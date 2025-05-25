import torch
import torch.nn as nn

# First, define the model class (same as in train.py)
class TransformerDiagnosticAgent(nn.Module):
    def __init__(self, input_dim=10, model_dim=128, num_heads=8, num_layers=3, num_actions=10, dropout=0.2):
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
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Load the model correctly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model instance
model = TransformerDiagnosticAgent(
    input_dim=10,
    model_dim=128,
    num_heads=8,
    num_layers=3,
    num_actions=10,
    dropout=0.2
)

# Load the saved state dict
checkpoint = torch.load("best_model.pth", map_location=device)
if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Correct input shape: (batch_size, sequence_length, input_features)
dummy_input = torch.randn(1, 20, 10)  # batch=1, seq_len=20, features=10

try:
    torch.onnx.export(
        model,
        dummy_input,
        "transformer_diagnostic_agent.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['q_values'],
        dynamic_axes={'input': {0: 'batch_size'}, 'q_values': {0: 'batch_size'}}
    )
    print("ONNX model exported successfully!")
    print("Upload 'transformer_diagnostic_agent.onnx' to https://netron.app for visualization")
except Exception as e:
    print(f"Error exporting ONNX: {e}")