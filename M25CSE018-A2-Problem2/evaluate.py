"""
evaluate.py - TASK-2: Quantitative Evaluation
Computes Novelty Rate and Diversity for each model's generated names.

Run from nlu_assignment/ directory:
    python evaluate.py
"""

import os, numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Resolve paths relative to this script
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_names.txt')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'outputs')

# ── Data loading ──────────────────────────────────────────────────────────────

def load_list(path):
    """Load strings from file, one per line, strip whitespace."""
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

# Build a set of training names in lowercase for fast O(1) membership tests
training_set = set(n.lower() for n in load_list(DATA_PATH))

# Map human-readable model names to their output files in outputs/
model_files = {
    'Vanilla RNN'   : 'vanilla_rnn_names.txt',
    'BLSTM'         : 'blstm_names.txt',
    'RNN+Attention' : 'rnn_attention_names.txt',
}

# ── Evaluation Metrics ────────────────────────────────────────────────────────

def novelty_rate(generated, training):
    """Novelty = proportion of generated names not seen in training data.

    Higher novelty means the model is not just memorising; it generalises
    to produce new, unseen character combinations.
    """
    novel = sum(1 for n in generated if n.lower() not in training)
    return novel / len(generated) if generated else 0.0

def diversity(generated):
    """Diversity = ratio of unique names to total generated.

    A diversity of 1.0 means every generated name is unique; lower values
    indicate the model is repeating the same names (mode collapse).
    """
    return len(set(generated)) / len(generated) if generated else 0.0

def avg_length(generated):
    """Average character length of generated names."""
    lengths = [len(n) for n in generated if n]
    return np.mean(lengths) if lengths else 0.0

def valid_rate(generated):
    """Plausibility heuristic: proportion of names with 2-15 characters.

    Very short names (< 2 chars) are likely degenerate outputs; very long
    names (> 15 chars) are implausibly long for personal names.
    """
    valid = sum(1 for n in generated if 2 <= len(n) <= 15)
    return valid / len(generated) if generated else 0.0

# ── Per-model Evaluation ──────────────────────────────────────────────────────

results = {}
print(f"\n{'='*70}")
print("TASK-2: QUANTITATIVE EVALUATION")
print(f"{'='*70}")
print(f"Training set size : {len(training_set)} names")
print()

# Print a formatted table header
header = f"{'Model':<18} {'N_gen':>6} {'Novelty%':>9} {'Diversity%':>11} {'ValidLen%':>10} {'AvgLen':>7}"
print(header)
print('-' * len(header))

for model_name, fname in model_files.items():
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        # Warn if a model's output file is missing (run train_all.py first)
        print(f"  {model_name}: FILE NOT FOUND ({path})")
        continue

    generated = load_list(path)

    # Compute all four metrics for this model
    nov  = novelty_rate(generated, training_set)
    div  = diversity(generated)
    vr   = valid_rate(generated)
    avgl = avg_length(generated)

    # Store results for later use in the sample and novel-names sections
    results[model_name] = dict(n=len(generated), novelty=nov, diversity=div,
                                valid=vr, avg_len=avgl, names=generated)

    # Print formatted row for this model
    print(f"{model_name:<18} {len(generated):>6} {nov*100:>8.1f}% {div*100:>10.1f}% "
          f"{vr*100:>9.1f}% {avgl:>7.2f}")

# ── Training Loss Analysis + Plot ─────────────────────────────────────────────

print(f"\n{'='*70}")
print("TRAINING LOSS (from training_losses.csv)")
print(f"{'='*70}")

csv_path = os.path.join(OUT_DIR, 'training_losses.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # Print loss reduction summary for each model
    for col, label in [('vanilla_rnn_loss', 'Vanilla RNN'),
                       ('blstm_loss',        'BLSTM'),
                       ('rnn_attention_loss','RNN+Attention')]:
        initial   = df[col].iloc[0]
        final     = df[col].iloc[-1]
        reduction = (initial - final) / initial * 100   # Percentage improvement
        print(f"  {label:<18}  epochs={len(df)}  "
              f"initial={initial:.4f}  final={final:.4f}  "
              f"reduction={reduction:.1f}%")

    # Plot training loss curves for all three models on the same axes
    plt.figure(figsize=(8, 4.5))
    plt.plot(df['epoch'], df['vanilla_rnn_loss'],   label='Vanilla RNN',     linewidth=2)
    plt.plot(df['epoch'], df['blstm_loss'],          label='BLSTM',           linewidth=2)
    plt.plot(df['epoch'], df['rnn_attention_loss'],  label='RNN + Attention', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.tight_layout()

    # Save the figure to the outputs directory
    plot_path = os.path.join(OUT_DIR, 'loss_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\n  Plot saved to {plot_path}")
else:
    # The CSV is created by train_all.py; remind the user to run it first
    print("  training_losses.csv not found -- run train_all.py first.")

# ── Representative Sample Outputs ─────────────────────────────────────────────

print(f"\n{'='*70}")
print("REPRESENTATIVE GENERATED SAMPLES (first 25 per model)")
print(f"{'='*70}")

for model_name, data in results.items():
    names = data['names']
    print(f"\n  {model_name}:")
    sample = names[:25]
    # Display in rows of 5 for a compact, readable layout
    for i in range(0, len(sample), 5):
        row = sample[i:i+5]
        print("    " + "  ".join(f"{n:<14}" for n in row))

# ── Novel Names Only ──────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("NOVEL NAMES NOT IN TRAINING SET (first 10 per model)")
print(f"{'='*70}")

for model_name, data in results.items():
    # Filter to only names the model invented rather than memorised
    novel = [n for n in data['names'] if n.lower() not in training_set]
    print(f"\n  {model_name}: {novel[:10]}")

print(f"\n{'='*70}")
print("Evaluation complete.")
