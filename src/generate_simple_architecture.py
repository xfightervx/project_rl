import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_simple_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define colors
    input_color = '#E3F2FD'
    transformer_color = '#FFECB3'
    attention_color = '#F3E5F5'
    qlearning_color = '#E8F5E8'
    output_color = '#FFEBEE'
    
    # Define box positions and sizes
    boxes = [
        # (x, y, width, height, label, color)
        (1, 3, 2, 1.5, 'Input Layer\n(Patient Data)\n10 Features', input_color),
        (4, 3, 2.5, 1.5, 'Transformer\nEncoder\n(3 Layers)', transformer_color),
        (7, 3, 2, 1.5, 'Attention\nMechanism', attention_color),
        (10, 2, 2, 1, 'Value Stream', qlearning_color),
        (10, 4, 2, 1, 'Advantage Stream', qlearning_color),
        (13, 3, 2, 1.5, 'Q-Values\nOutput\n(10 Actions)', output_color),
    ]
    
    # Draw boxes
    for x, y, w, h, label, color in boxes:
        box = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor='black',
                           linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, 
               ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        # (start_x, start_y, end_x, end_y)
        (3, 3.75, 4, 3.75),      # Input -> Transformer
        (6.5, 3.75, 7, 3.75),    # Transformer -> Attention
        (9, 3.75, 10, 2.5),      # Attention -> Value
        (9, 3.75, 10, 4.5),      # Attention -> Advantage
        (12, 2.5, 13, 3.25),     # Value -> Output
        (12, 4.5, 13, 4.25),     # Advantage -> Output
    ]
    
    for start_x, start_y, end_x, end_y in arrows:
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add main title
    ax.text(8, 6, 'Transformer-based DQN Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add component descriptions
    descriptions = [
        (2, 1.5, 'Patient vitals,\nsubjective symptoms,\ntest results'),
        (5.25, 1.5, 'Multi-head attention\nwith positional encoding'),
        (8, 1.5, 'Weighted feature\naggregation'),
        (11, 1, 'State value\nestimation'),
        (11, 5.5, 'Action advantage\nestimation'),
        (14, 1.5, 'Q(s,a) = V(s) +\nA(s,a) - mean(A)')
    ]
    
    for x, y, desc in descriptions:
        ax.text(x, y, desc, ha='center', va='center', 
               fontsize=8, style='italic', color='gray')
    
    # Set axis properties
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('simple_architecture.pdf', bbox_inches='tight')
    plt.show()

def create_text_overview():
    overview = """
    TRANSFORMER-BASED DQN ARCHITECTURE OVERVIEW
    ==========================================
    
    1. INPUT LAYER
       - 10 features: vitals + subjective symptoms + test results
       - Sequence length: 20 timesteps
       - Shape: (batch_size, 20, 10)
    
    2. TRANSFORMER ENCODER
       - 3 encoder layers
       - 8 attention heads
       - Model dimension: 128
       - Dropout: 0.2
    
    3. ATTENTION MECHANISM
       - Weighted aggregation of sequence features
       - Context vector generation
       - Temporal pattern recognition
    
    4. DUELING Q-NETWORK
       - Value Stream: Estimates state value V(s)
       - Advantage Stream: Estimates action advantages A(s,a)
       - Output: Q(s,a) = V(s) + A(s,a) - mean(A)
    
    5. OUTPUT
       - 10 Q-values (one per action)
       - Actions: Wait, Test, Alert, Diagnose (6 diseases)
    
    TRAINING METHOD: Deep Q-Learning with Experience Replay
    """
    print(overview)
    
    # Save to file
    with open('architecture_overview.txt', 'w') as f:
        f.write(overview)

if __name__ == "__main__":
    create_simple_architecture_diagram()
    create_text_overview()
    print("Simple architecture diagram saved as 'simple_architecture.png'")
    print("Text overview saved as 'architecture_overview.txt'")