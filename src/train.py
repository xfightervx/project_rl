import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from collections import deque
import random
import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

train_losses = []
train_rewards = []
train_f1s = []
val_f1s = []
val_rewards = []

# ------------------- Dataset Class ------------------- #
class PatientDataset(Dataset):
    def __init__(self, csv_path, seq_len=20):
        df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.person_ids = df['person_id'].unique()
        self.feature_cols = [
            'heart_rate', 'blood_pressure', 'oxygen_saturation', 'temperature',
            'breathing_quality', 'soreness', 'fatigue', 'mental_clarity', 'appetite',
            'test_result'  # NEW: test result signal
        ]
        self.label_map = {d: i for i, d in enumerate(sorted(df['true_disease'].unique()))}
        self.data = df

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        pid = self.person_ids[idx]
        person_df = self.data[self.data['person_id'] == pid].sort_values(by='timestep')
        
        # Convert string dictionaries to actual dictionaries and extract features
        vitals_df = pd.DataFrame([eval(v) for v in person_df['vitals']])
        subjective_df = pd.DataFrame([eval(s) for s in person_df['subjective']])
        
        # Combine all features
        features_df = pd.concat([
            vitals_df[['heart_rate', 'blood_pressure', 'oxygen_saturation', 'temperature']],
            subjective_df[['breathing_quality', 'soreness', 'fatigue', 'mental_clarity', 'appetite']]
        ], axis=1)
        
        features = features_df.values.astype(np.float32)
        
        # Add test_result column if missing
        test_result_col = np.zeros((features.shape[0], 1), dtype=np.float32)
        features = np.hstack((features, test_result_col))
        
        label = person_df['label'].iloc[0]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class PrioritizedExperienceBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences in buffer. Have {len(self.buffer)}, need {batch_size}")
            
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Convert to tuple of batches instead of separate lists
        return tuple(zip(*samples))

    def __len__(self):
        return len(self.buffer)

# ------------------- Transformer Model ------------------- #
class TransformerDiagnosticAgent(nn.Module):
    def __init__(self, input_dim=10, model_dim=64, num_heads=4, num_layers=2, num_actions=10, dropout=0.1):
        super(TransformerDiagnosticAgent, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.bn1 = nn.BatchNorm1d(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attn_layer = nn.Linear(model_dim, 1)
        
        # Separate advantage and value streams
        self.value_stream = nn.Sequential(
            nn.Linear(model_dim + input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(model_dim + input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_proj = self.input_proj(x)
        x_proj = self.bn1(x_proj.transpose(1, 2)).transpose(1, 2)
        
        enc_out = self.transformer_encoder(x_proj)
        attn_weights = torch.softmax(self.attn_layer(enc_out), dim=1)
        context_vector = torch.sum(attn_weights * enc_out, dim=1)
        
        last_input = x[:, -1, :]
        combined = torch.cat([context_vector, last_input], dim=1)
        
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        
        # Q-value computation using dueling architecture
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ------------------- Reward Function ------------------- #
def compute_reward(action, label, is_sick, used_general_test, used_detailed_test):
    if action == 0:  # Wait
        return 0.0 if is_sick else 0.5  # Small positive reward for correct waiting
    elif action == 1:  # General Test
        if used_general_test:
            return -2.0  # Reduced penalty
        return 2.0 if is_sick else -1.0  # Increased reward for appropriate testing
    elif action == 2:  # Alert
        return 15.0 if is_sick else -5.0  # Higher reward for correct alert
    elif 3 <= action <= 8:  # Diagnosis
        correct_diagnosis = (action - 3 == label)
        if correct_diagnosis:
            return 25.0  # Higher reward for correct diagnosis
        return -5.0  # Reduced penalty
    elif action == 9:  # Detailed Test
        if not used_general_test:
            return -2.0
        return 3.0 if is_sick else -1.0
    return -1.0  # Reduced default penalty

def normalize_rewards(rewards):
    # Use clone().detach() instead of torch.tensor()
    rewards = rewards.clone().detach()
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# ------------------- Q-learning Training ------------------- #
def train_q_learning(model, train_loader, val_loader, num_epochs=20, gamma=0.99, lr=1e-4,
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Reduce learning rate and add weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Use Huber loss instead of MSE for better stability
    criterion = nn.SmoothL1Loss()
    
    # Use prioritized experience replay
    exp_buffer = PrioritizedExperienceBuffer(capacity=50000)
    
    # Modify epsilon parameters
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    # Add target network
    target_model = TransformerDiagnosticAgent(
        input_dim=10,
        model_dim=128,
        num_heads=8,
        num_layers=3,
        num_actions=10,
        dropout=0.2
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # Update target network periodically
    target_update_freq = 100
    steps = 0
    
    # Calculate total steps for progress bar
    total_steps = num_epochs * len(train_loader)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Early stopping variables
    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    # Create progress bar
    pbar = tqdm(total=total_steps, desc='Training Progress')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        episode_rewards = []

        # Add description to show current epoch
        pbar.set_description(f'Epoch {epoch+1}/{num_epochs}')

        for state, label in train_loader:
            state = state.to(device)
            label = label.to(device)
            batch_size = state.size(0)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                predicted_actions = torch.randint(0, 10, (batch_size,)).to(device)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    predicted_actions = torch.argmax(q_values, dim=1)

            used_general_test = torch.zeros(batch_size, dtype=torch.bool).to(device)
            used_detailed_test = torch.zeros(batch_size, dtype=torch.bool).to(device)
            final_actions = predicted_actions.clone()
            rewards = []

            # Process actions and collect rewards
            for i in range(batch_size):
                action = predicted_actions[i].item()
                is_sick = label[i].item() != 0
                
                if action == 1:  # General Test
                    if used_general_test[i]:
                        reward = -10.0
                    else:
                        used_general_test[i] = True
                        if is_sick and random.random() < 0.95:
                            state[i, :, 0] += 0.1
                        elif not is_sick and random.random() < 0.05:
                            state[i, :, 0] += 0.1
                        with torch.no_grad():
                            q_values = model(state)
                        action = torch.argmax(q_values[i]).item()
                        final_actions[i] = action
                        reward = compute_reward(action, label[i].item(), is_sick, True, False)
                
                elif action == 9:  # Detailed Test
                    if used_detailed_test[i] or not used_general_test[i]:
                        reward = -10.0
                    else:
                        used_detailed_test[i] = True
                        boost = (epoch + 1) / 10.0 + 0.3
                        state[i, :, 1] += boost
                        with torch.no_grad():
                            q_values = model(state)
                        action = torch.argmax(q_values[i]).item()
                        final_actions[i] = action
                        reward = compute_reward(action, label[i].item(), is_sick, True, True)
                
                else:
                    reward = compute_reward(action, label[i].item(), is_sick, 
                                         used_general_test[i], used_detailed_test[i])
                
                rewards.append(reward)
                exp_buffer.push(state[i].cpu(), action, reward, 
                              state[i].cpu())  # Store experience

            rewards = torch.FloatTensor(rewards).to(device)  # If rewards is a list
            # OR
            # rewards = torch.stack(rewards).to(device)  # If rewards contains tensors
            rewards = normalize_rewards(rewards).to(device)
            
            # Training step
            if len(exp_buffer) >= batch_size:
                states, actions, rewards, next_states = exp_buffer.sample(batch_size)
                
                states = torch.stack(states).to(device)
                next_states = torch.stack(next_states).to(device)
                actions = torch.tensor(actions).to(device)
                rewards = torch.tensor(rewards).to(device)

                current_q_values = model(states)
                next_q_values = target_model(next_states).detach()
                
                # Compute target Q-values with double Q-learning
                next_actions = model(next_states).argmax(dim=1)
                max_next_q = next_q_values[torch.arange(batch_size), next_actions]
                
                # Add n-step returns (n=3)
                n_step_reward = rewards
                for i in range(2):  # Additional 2 steps
                    if i < len(rewards) - 1:
                        n_step_reward += (gamma ** (i + 1)) * rewards[i + 1]
                
                target_q = current_q_values.clone()
                for i in range(batch_size):
                    target_q[i, actions[i]] = n_step_reward[i] + (gamma ** 3) * max_next_q[i]

                loss = criterion(current_q_values, target_q)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            all_preds.extend(final_actions.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
            episode_rewards.extend(rewards.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_reward = np.mean(episode_rewards)
        train_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Validation phase
        val_f1, val_reward = validate_model(model, val_loader, device)
        
        # Store metrics
        train_losses.append(epoch_loss)
        train_rewards.append(epoch_reward)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        val_rewards.append(val_reward)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Reward/train', epoch_reward, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Reward/val', val_reward, epoch)
        
        # Early stopping check
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, "
                   f"Mean Reward: {epoch_reward:.2f}, F1 Score: {train_f1:.4f}, "
                   f"Epsilon: {epsilon:.4f}")

    pbar.close()
    writer.close()
    return model, {
        'train_losses': train_losses,
        'train_rewards': train_rewards,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'val_rewards': val_rewards
    }

def validate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_reward = 0
    
    # Add progress bar for validation
    val_pbar = tqdm(dataloader, desc='Validating')
    
    with torch.no_grad():
        for state, label in val_pbar:
            state = state.to(device)
            label = label.to(device)
            
            q_values = model(state)
            actions = torch.argmax(q_values, dim=1)
            
            rewards = []
            for i in range(len(actions)):
                is_sick = label[i].item() != 0
                reward = compute_reward(actions[i].item(), label[i].item(), 
                                     is_sick, False, False)
                rewards.append(reward)
            
            total_reward += sum(rewards)
            all_preds.extend(actions.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
            
            # Update progress bar with current metrics
            val_pbar.set_postfix({
                'reward': f'{total_reward/len(all_preds):.2f}'
            })
    
    val_pbar.close()
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return f1, total_reward / len(dataloader.dataset)

def plot_metrics(metrics):
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot F1 scores
    plt.subplot(2, 2, 2)
    plt.plot(metrics['train_f1s'], label='Train')
    plt.plot(metrics['val_f1s'], label='Validation')
    plt.title('F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot rewards
    plt.subplot(2, 2, 3)
    plt.plot(metrics['train_rewards'], label='Train')
    plt.plot(metrics['val_rewards'], label='Validation')
    plt.title('Average Rewards')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == "__main__":
    # Load and split the dataset
    train_dataset = PatientDataset("output/train.csv")
    val_dataset = PatientDataset("output/val.csv")
    test_dataset = PatientDataset("output/test.csv")
    dataset = train_dataset
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train model
    model = TransformerDiagnosticAgent(
        input_dim=10,
        model_dim=128,  
        num_heads=8,   
        num_layers=3,   
        num_actions=10,
        dropout=0.2     
    )
    trained_model, metrics = train_q_learning(model, train_loader, val_loader)
    
    # Plot training metrics
    plot_metrics(metrics)
    
    # Final evaluation on test set
    test_f1, test_reward = validate_model(trained_model, test_loader, 
                                        'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Test Set Performance - F1 Score: {test_f1:.4f}, "
               f"Average Reward: {test_reward:.2f}")
    
    # TensorBoard test logging
    writer = SummaryWriter(log_dir="runs/test")
    # Use the writer to log data
    writer.add_scalar('Loss/train', 0.5, 0)
    writer.close()

