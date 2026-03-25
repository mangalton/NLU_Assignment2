"""
train_all.py — Train all three models and save generated names.
Run from the nlu_assignment/ directory:
    python train_all.py
"""

import sys, os, time, csv
import numpy as np

# Add models directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from vanilla_rnn   import VanillaRNN, load_names, build_vocab
from bilstm         import BLSTM
from rnn_attention import RNNAttention

# Setup paths and data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_names.txt')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# Load vocabulary once and reuse across all models
names  = load_names(DATA_PATH)
ch2id, id2ch = build_vocab(names)
V = len(ch2id)

# Common hyperparameters for fair comparison
EPOCHS   = 150
LR       = 0.005
N_GEN    = 200
TEMP     = 0.8
SEED     = 42

stats = {}  # Store metrics for summary table

# ─────────────────────── Vanilla RNN ───────────────────────────────────────
print("\n" + "="*60)
print("MODEL 1: Vanilla RNN")
print("="*60)
rnn = VanillaRNN(vocab_size=V, hidden_size=128, seed=SEED)
print(f"  Trainable params : {rnn.count_params():,}")
t0 = time.time()
rnn_losses = rnn.train(names, ch2id, epochs=EPOCHS, lr=LR, print_every=30)
rnn_time   = time.time() - t0
rnn_names  = rnn.generate(ch2id, id2ch, n=N_GEN, temperature=TEMP, seed=SEED)
with open(os.path.join(OUT_DIR, 'vanilla_rnn_names.txt'), 'w') as f:
    f.write('\n'.join(rnn_names))
stats['Vanilla RNN'] = {
    'params': rnn.count_params(),
    'final_loss': rnn_losses[-1],
    'time_s': rnn_time,
    'generated': rnn_names
}
print(f"  Training time: {rnn_time:.1f}s  |  Final loss: {rnn_losses[-1]:.4f}")
print(f"  Sample: {rnn_names[:8]}")
# Save model weights for later use
rnn.save(os.path.join(OUT_DIR, 'vanilla_rnn'))        

# ─────────────────────── BLSTM ──────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 2: Bidirectional LSTM")
print("="*60)
# Smaller hidden size (96 vs 128) to keep parameter count comparable
blstm = BLSTM(vocab_size=V, hidden_size=96, seed=SEED)
print(f"  Trainable params : {blstm.count_params():,}")
t0 = time.time()
blstm_losses = blstm.train(names, ch2id, epochs=EPOCHS, lr=LR, print_every=30)
blstm_time   = time.time() - t0
blstm_names  = blstm.generate(ch2id, id2ch, n=N_GEN, temperature=TEMP, seed=SEED)
with open(os.path.join(OUT_DIR, 'blstm_names.txt'), 'w') as f:
    f.write('\n'.join(blstm_names))
stats['BLSTM'] = {
    'params': blstm.count_params(),
    'final_loss': blstm_losses[-1],
    'time_s': blstm_time,
    'generated': blstm_names
}
print(f"  Training time: {blstm_time:.1f}s  |  Final loss: {blstm_losses[-1]:.4f}")
print(f"  Sample: {blstm_names[:8]}")



# ─────────────────────── RNN + Attention ────────────────────────────────────
print("\n" + "="*60)
print("MODEL 3: RNN + Attention")
print("="*60)
# Adds attention mechanism to standard RNN
attn = RNNAttention(vocab_size=V, hidden_size=128, attn_size=64, seed=SEED)
print(f"  Trainable params : {attn.count_params():,}")
t0 = time.time()
attn_losses = attn.train(names, ch2id, epochs=EPOCHS, lr=LR, print_every=30)
attn_time   = time.time() - t0
attn_names  = attn.generate(ch2id, id2ch, n=N_GEN, temperature=TEMP, seed=SEED)
with open(os.path.join(OUT_DIR, 'rnn_attention_names.txt'), 'w') as f:
    f.write('\n'.join(attn_names))
stats['RNN+Attention'] = {
    'params': attn.count_params(),
    'final_loss': attn_losses[-1],
    'time_s': attn_time,
    'generated': attn_names
}
print(f"  Training time: {attn_time:.1f}s  |  Final loss: {attn_losses[-1]:.4f}")
print(f"  Sample: {attn_names[:8]}")


# ─────────────────────── Summary ─────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"{'Model':<20} {'Params':>10} {'Loss':>8} {'Time(s)':>10}")
print("-"*52)
for m, s in stats.items():
    print(f"{m:<20} {s['params']:>10,} {s['final_loss']:>8.4f} {s['time_s']:>10.1f}")

# Save loss curves as CSV for visualization
csv_path = os.path.join(OUT_DIR, 'training_losses.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'vanilla_rnn_loss', 'blstm_loss', 'rnn_attention_loss'])
    # Combine losses from all three models for each epoch
    for i, (r, b, a) in enumerate(zip(rnn_losses, blstm_losses, attn_losses), start=1):
        writer.writerow([i, round(float(r), 6), round(float(b), 6), round(float(a), 6)])
print(f"\nLoss curves saved to {csv_path}")
print("All generated names saved to outputs/")