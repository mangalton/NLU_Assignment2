"""
TASK-1: Vanilla RNN for Character-Level Name Generation
Implemented from scratch using NumPy (no deep learning frameworks).

Architecture:
  - Embedding layer: one-hot encoded characters
  - Single hidden RNN layer with tanh activation
  - Output linear layer with softmax
"""

from xml.parsers.expat import model

import numpy as np
import os
import sys

# ─────────────────────────── Data Utilities ────────────────────────────

def load_names(path):
    """Read names from file, convert to lowercase, strip whitespace."""
    with open(path) as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names

def build_vocab(names):
    """Create character mappings with start (^) and end ($) tokens."""
    chars = sorted(set(''.join(names)))
    chars = ['^'] + chars + ['$']   # ^ = start, $ = end
    ch2id = {c: i for i, c in enumerate(chars)}
    id2ch = {i: c for c, i in ch2id.items()}
    return ch2id, id2ch

def name_to_seqs(name, ch2id):
    """Convert name to input/target sequences using start/end tokens."""
    seq = [ch2id['^']] + [ch2id[c] for c in name] + [ch2id['$']]
    return seq[:-1], seq[1:]  # inputs shifted by 1 relative to targets


# ──────────────────────────── Vanilla RNN ───────────────────────────────

class VanillaRNN:
    """
    Single-layer Vanilla RNN.

    Hyperparameters
    ---------------
    hidden_size : 128
    layers      : 1
    learning_rate: 0.005 (with gradient clipping at 5.0)
    """

    def __init__(self, vocab_size, hidden_size=128, seed=42):
        np.random.seed(seed)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size

        # Xavier initialization to prevent vanishing/exploding gradients
        scale_xh = np.sqrt(2.0 / (vocab_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2.0 / (hidden_size + vocab_size))

        self.Wxh = np.random.randn(hidden_size, vocab_size)  * scale_xh  # input→hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale_hh  # hidden→hidden
        self.bh  = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(vocab_size, hidden_size)  * scale_hy  # hidden→output
        self.by  = np.zeros((vocab_size, 1))

        # Adagrad accumulators for adaptive learning rates
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mbh  = np.zeros_like(self.bh)
        self.mWhy = np.zeros_like(self.Why)
        self.mby  = np.zeros_like(self.by)

    def count_params(self):
        return (self.Wxh.size + self.Whh.size + self.bh.size +
                self.Why.size + self.by.size)

    def forward(self, inputs, h_prev):
        """
        Forward pass with caching for BPTT.
        Stores all intermediate states to enable backprop through time.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev.copy()
        loss = 0.0
        T = len(inputs)
        for t in range(T - 1):
            # One-hot encode current character
            x = np.zeros((self.vocab_size, 1))
            x[inputs[t]] = 1
            xs[t] = x
            
            # Recurrent update: new hidden = tanh(Wxh·x + Whh·h_prev + bh)
            hs[t] = np.tanh(self.Wxh @ x + self.Whh @ hs[t-1] + self.bh)
            
            # Output layer and softmax
            ys[t] = self.Why @ hs[t] + self.by
            e = np.exp(ys[t] - ys[t].max())  # subtract max for numerical stability
            ps[t] = e / e.sum()
            
            # Cross-entropy loss with next character as target
            loss -= np.log(ps[t][inputs[t+1], 0] + 1e-12)
        return xs, hs, ys, ps, loss

    def backward(self, inputs, xs, hs, ps, clip=5.0):
        """Backprop through time with gradient clipping."""
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh  = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby  = np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))

        T = len(inputs)
        for t in reversed(range(T - 1)):  # Iterate backwards through time
            # Gradient from output: softmax + cross-entropy simplified
            dy = ps[t].copy()
            dy[inputs[t+1]] -= 1  # ∂L/∂y = softmax - one_hot(target)
            
            dWhy += dy @ hs[t].T
            dby  += dy
            
            # Backprop through hidden layer
            dh = self.Why.T @ dy + dh_next
            dhraw = (1 - hs[t]**2) * dh  # tanh derivative
            
            dbh  += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t-1].T
            dh_next = self.Whh.T @ dhraw

        # Prevent exploding gradients
        for d in [dWxh, dWhh, dbh, dWhy, dby]:
            np.clip(d, -clip, clip, out=d)

        return dWxh, dWhh, dbh, dWhy, dby

    def adagrad_update(self, grads, lr=0.005):
        """Update parameters using Adagrad (per-parameter adaptive learning rates)."""
        dWxh, dWhh, dbh, dWhy, dby = grads
        eps = 1e-8
        for param, dparam, mem in [
            (self.Wxh, dWxh, self.mWxh),
            (self.Whh, dWhh, self.mWhh),
            (self.bh,  dbh,  self.mbh),
            (self.Why, dWhy, self.mWhy),
            (self.by,  dby,  self.mby),
        ]:
            mem += dparam * dparam  # accumulate squared gradients
            param -= lr * dparam / (np.sqrt(mem) + eps)  # adaptive step size

    def train(self, names, ch2id, epochs=150, lr=0.005, print_every=10):
        """Train the RNN using SGD with Adagrad over multiple epochs."""
        losses = []
        h_prev = np.zeros((self.hidden_size, 1))

        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(names)  # stochastic training
            for name in names:
                seq_in, seq_out = name_to_seqs(name, ch2id)
                full = seq_in + [seq_out[-1]]
                
                # Forward-backward pass for this sequence
                xs, hs, ys, ps, loss = self.forward(full, h_prev)
                epoch_loss += loss
                grads = self.backward(full, xs, hs, ps)
                self.adagrad_update(grads, lr=lr)
                
                h_prev = np.zeros((self.hidden_size, 1))  # Reset state per name

            avg = epoch_loss / len(names)
            losses.append(avg)
            if (epoch + 1) % print_every == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

        return losses

    def generate(self, ch2id, id2ch, n=200, max_len=20, temperature=0.8, seed=None):
        """
        Generate names by sampling from the probability distribution.
        Temperature controls diversity: lower = more conservative, higher = more random.
        """
        if seed is not None:
            np.random.seed(seed)
        generated = []
        for _ in range(n):
            h = np.zeros((self.hidden_size, 1))
            idx = ch2id['^']
            name_chars = []
            for _ in range(max_len):
                x = np.zeros((self.vocab_size, 1))
                x[idx] = 1
                h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
                y = self.Why @ h + self.by
                
                # Apply temperature scaling
                y = y / temperature
                e = np.exp(y - y.max())
                p = (e / e.sum()).ravel()
                
                idx = np.random.choice(self.vocab_size, p=p)
                if idx == ch2id['$']:  # Stop at end token
                    break
                name_chars.append(id2ch[idx])
            
            if name_chars:
                name = ''.join(name_chars).capitalize()
                generated.append(name)
        return generated
    
    def save(self, path):
        """Persist model weights to disk."""
        np.savez(path,
            Wxh=self.Wxh, Whh=self.Whh, bh=self.bh,
            Why=self.Why, by=self.by,
            vocab_size=self.vocab_size, hidden_size=self.hidden_size
        )

    @classmethod
    def load(cls, path):
        """Load model weights from disk."""
        d = np.load(path + '.npz')
        model = cls(vocab_size=int(d['vocab_size']), hidden_size=int(d['hidden_size']))
        model.Wxh = d['Wxh']; model.Whh = d['Whh']; model.bh = d['bh']
        model.Why = d['Why']; model.by  = d['by']
        return model


# ──────────────────────────── Main ──────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = os.path.join(os.path.dirname(__file__),'training_names.txt')
    names = load_names(DATA_PATH)
    ch2id, id2ch = build_vocab(names)
    vocab_size = len(ch2id)

    print(f"Vocab size: {vocab_size}, Training names: {len(names)}")

    model = VanillaRNN(vocab_size=vocab_size, hidden_size=128)
    print(f"Trainable parameters: {model.count_params():,}")
    print("Architecture: Input(one-hot) → Linear(Wxh) + Recurrent(Whh) → tanh → Linear(Why) → Softmax")
    print("Hyperparameters: hidden_size=128, layers=1, lr=0.005 (Adagrad), clip=5.0")
    print("\nTraining Vanilla RNN...")

    losses = model.train(names, ch2id, epochs=150, lr=0.005, print_every=15)

    generated = model.generate(ch2id, id2ch, n=200, seed=7)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'vanilla_rnn_names.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for nm in generated:
            f.write(nm + '\n')

    print(f"\nGenerated {len(generated)} names → {out_path}")
    print("Sample:", generated[:20])