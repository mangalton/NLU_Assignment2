"""
evaluate.py — TASK-2: Quantitative Evaluation
Computes Novelty Rate and Diversity for each model's generated names.

Run from nlu_assignment/ directory:
    python evaluate.py
"""

import os, numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_names.txt')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'outputs')

# ── Load data ────────────────────────────────────────────────────────────────
def load_list(path):
    """Load strings from file, one per line, strip whitespace."""
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

# Training set for novelty calculation (case-insensitive)
training_set = set(n.lower() for n in load_list(DATA_PATH))

# Map model names to their output files
model_files = {
    'Vanilla RNN'   : 'vanilla_rnn_names.txt',
    'BLSTM'         : 'blstm_names.txt',
    'RNN+Attention' : 'rnn_attention_names.txt',
}

# ── Metrics ──────────────────────────────────────────────────────────────────
def novelty_rate(generated, training):
    """Novelty = proportion of generated names not seen in training data.
    Higher values indicate better generalization/creativity."""
    novel = sum(1 for n in generated if n.lower() not in training)
    return novel / len(generated) if generated else 0.0

def diversity(generated):
    """Diversity = ratio of unique names to total generated.
    Higher values mean less repetition, more variety."""
    return len(set(generated)) / len(generated) if generated else 0.0

def avg_length(generated):
    """Average character length of generated names."""
    lengths = [len(n) for n in generated if n]
    return np.mean(lengths) if lengths else 0.0

def valid_rate(generated):
    """Plausibility heuristic: names with 2-15 characters.
    Filters out extremely short or implausibly long names."""
    valid = sum(1 for n in generated if 2 <= len(n) <= 15)
    return valid / len(generated) if generated else 0.0

# ── Evaluate each model ───────────────────────────────────────────────────────
results = {}
print(f"\n{'='*70}")
print("TASK-2: QUANTITATIVE EVALUATION")
print(f"{'='*70}")
print(f"Training set size : {len(training_set)} names")
print()

header = f"{'Model':<18} {'N_gen':>6} {'Novelty%':>9} {'Diversity%':>11} {'ValidLen%':>10} {'AvgLen':>7}"
print(header)
print('-' * len(header))

for model_name, fname in model_files.items():
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        print(f"  {model_name}: FILE NOT FOUND ({path})")
        continue
    generated = load_list(path)
    nov  = novelty_rate(generated, training_set)
    div  = diversity(generated)
    vr   = valid_rate(generated)
    avgl = avg_length(generated)
    results[model_name] = dict(n=len(generated), novelty=nov, diversity=div,
                                valid=vr, avg_len=avgl, names=generated)
    print(f"{model_name:<18} {len(generated):>6} {nov*100:>8.1f}% {div*100:>10.1f}% "
          f"{vr*100:>9.1f}% {avgl:>7.2f}")

# ── Training loss summary + plot ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("TRAINING LOSS (from training_losses.csv)")
print(f"{'='*70}")
csv_path = os.path.join(OUT_DIR, 'training_losses.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    # Compute loss reduction percentages for each model
    for col, label in [('vanilla_rnn_loss', 'Vanilla RNN'),
                       ('blstm_loss',        'BLSTM'),
                       ('rnn_attention_loss','RNN+Attention')]:
        initial = df[col].iloc[0]
        final   = df[col].iloc[-1]
        reduction = (initial - final) / initial * 100
        print(f"  {label:<18}  epochs={len(df)}  "
              f"initial={initial:.4f}  final={final:.4f}  "
              f"reduction={reduction:.1f}%")
 
    # Visualize training progress across all models
    plt.figure(figsize=(8, 4.5))
    plt.plot(df['epoch'], df['vanilla_rnn_loss'],   label='Vanilla RNN',    linewidth=2)
    plt.plot(df['epoch'], df['blstm_loss'],          label='BLSTM',          linewidth=2)
    plt.plot(df['epoch'], df['rnn_attention_loss'],  label='RNN + Attention', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'loss_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\n  Plot saved to {plot_path}")
else:
    print("  training_losses.csv not found — run train_all.py first.")
 
# ── Sample outputs ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("REPRESENTATIVE GENERATED SAMPLES (first 25 per model)")
print(f"{'='*70}")
for model_name, data in results.items():
    names = data['names']
    print(f"\n  {model_name}:")
    sample = names[:25]
    # Display in rows of 5 for compact visualization
    for i in range(0, len(sample), 5):
        row = sample[i:i+5]
        print("    " + "  ".join(f"{n:<14}" for n in row))

# ── Novel names only ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("NOVEL NAMES NOT IN TRAINING SET (first 10 per model)")
print(f"{'='*70}")
for model_name, data in results.items():
    # Filter to only show names the model invented (not memorized)
    novel = [n for n in data['names'] if n.lower() not in training_set]
    print(f"\n  {model_name}: {novel[:10]}")

print(f"\n{'='*70}")
print("Evaluation complete.")