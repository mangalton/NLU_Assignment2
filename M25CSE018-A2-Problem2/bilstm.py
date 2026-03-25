"""
TASK-1: Bidirectional LSTM (BLSTM) for Character-Level Name Generation
Implemented from scratch using NumPy (no deep learning frameworks).

Architecture
------------
Training:
  - Forward LSTM  (left→right) over the input prefix
  - Backward LSTM (right→left) over the reversed prefix
  - Concatenated hidden states [h_fwd || h_bwd] fed to a COMBINED output head
  - Forward hidden states h_fwd also fed to a FORWARD-ONLY output head
    (trained with auxiliary weight 0.5 so it learns a standalone LM)

Generation (autoregressive — cannot use backward direction):
  - Forward LSTM only, using the FORWARD-ONLY output head
  - No distribution mismatch: this head was trained purely on forward states

This dual-head design is the standard solution for BLSTM language modelling
(cf. Sundermeyer et al., 2014; Peters et al. ELMo, 2018).

Hyperparameters
---------------
hidden_size   : 96 per direction  (192 after concatenation)
layers        : 1
learning_rate : 0.003  (Adagrad)
aux_weight    : 0.5    (forward-only head loss weight)
gradient clip : 5.0
temperature   : 0.85   (generation)
"""

import numpy as np
import os


# ─────────────────────── utilities ───────────────────────────────────────────

def load_names(path):
    with open(path) as f:
        return [line.strip().lower() for line in f if line.strip()]

def build_vocab(names):
    chars = sorted(set(''.join(names)))
    chars = ['^'] + chars + ['$']
    ch2id = {c: i for i, c in enumerate(chars)}
    id2ch = {i: c for c, i in ch2id.items()}
    return ch2id, id2ch

def sigmoid(x):
    """Numerically stable sigmoid: handles large positive/negative values."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def one_hot(idx, size):
    x = np.zeros((size, 1))
    x[idx] = 1.0
    return x



# ─────────────────────── LSTM cell ───────────────────────────────────────────

class LSTMCell:
    """Single-direction LSTM layer with packed gate matrices."""

    def __init__(self, input_size, hidden_size, seed=0):
        np.random.seed(seed)
        H, I = hidden_size, input_size
        s = np.sqrt(2.0 / (I + H))  # Xavier initialization
        # Pack all four gates [f, i, g, o] into one matrix each for efficiency
        self.Wx = np.random.randn(4 * H, I) * s
        self.Wh = np.random.randn(4 * H, H) * s
        self.b  = np.zeros((4 * H, 1))
        self.H  = H
        # Adagrad accumulators
        self.mWx = np.zeros_like(self.Wx)
        self.mWh = np.zeros_like(self.Wh)
        self.mb  = np.zeros_like(self.b)

    def count_params(self):
        return self.Wx.size + self.Wh.size + self.b.size

    def step(self, x, h_prev, c_prev):
        """Single LSTM step: computes new hidden and cell states.
        Returns: h, c, cache (for BPTT)"""
        H = self.H
        z = self.Wx @ x + self.Wh @ h_prev + self.b   # (4H, 1)
        
        # Split into four gates and apply activations
        f = sigmoid(z[:H])      # forget gate
        i = sigmoid(z[H:2*H])   # input gate
        g = np.tanh(z[2*H:3*H]) # candidate cell state
        o = sigmoid(z[3*H:])    # output gate
        
        c = f * c_prev + i * g   # new cell state
        h = o * np.tanh(c)       # new hidden state
        return h, c, (x, h_prev, c_prev, f, i, g, o, c)

    def forward_seq(self, xs):
        """Process a sequence of inputs through LSTM."""
        H  = self.H
        h  = np.zeros((H, 1))
        c  = np.zeros((H, 1))
        hs, cs, caches = [], [], []
        for x in xs:
            h, c, cache = self.step(x, h, c)
            hs.append(h)
            cs.append(c)
            caches.append(cache)
        return hs, cs, caches

    def backward_seq(self, caches, dh_list, dc_list, clip=5.0):
        """
        BPTT over full sequence.
        dh_list[t]: gradient into hidden state at time t
        dc_list[t]: gradient into cell state at time t (usually zeros)
        Returns gradients for Wx, Wh, b.
        """
        H   = self.H
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db  = np.zeros_like(self.b)
        dh_next = np.zeros((H, 1))
        dc_next = np.zeros((H, 1))

        # Backward through time
        for t in reversed(range(len(caches))):
            x, h_prev, c_prev, f, i, g, o, c = caches[t]
            dh = dh_list[t] + dh_next
            dc = dc_list[t] + dc_next

            # Gradients through LSTM gates (chain rule)
            do = dh * np.tanh(c)
            dc = dc + dh * o * (1.0 - np.tanh(c) ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i

            # Pre-activation gradients (with derivative of sigmoid/tanh)
            dz = np.vstack([
                df * f * (1.0 - f),      # forget gate
                di * i * (1.0 - i),      # input gate
                dg * (1.0 - g ** 2),     # candidate cell
                do * o * (1.0 - o),      # output gate
            ])
            
            dWx     += dz @ x.T
            dWh     += dz @ h_prev.T
            db      += dz
            dh_next  = self.Wh.T @ dz
            dc_next  = dc * f

        # Prevent exploding gradients
        np.clip(dWx, -clip, clip, out=dWx)
        np.clip(dWh, -clip, clip, out=dWh)
        np.clip(db,  -clip, clip, out=db)
        return dWx, dWh, db

    def adagrad_step(self, dWx, dWh, db, lr):
        """Adagrad update with parameter-specific learning rates."""
        eps = 1e-8
        self.mWx += dWx ** 2;  self.Wx -= lr * dWx / (np.sqrt(self.mWx) + eps)
        self.mWh += dWh ** 2;  self.Wh -= lr * dWh / (np.sqrt(self.mWh) + eps)
        self.mb  += db  ** 2;  self.b  -= lr * db  / (np.sqrt(self.mb)  + eps)


# ─────────────────────── BLSTM model ─────────────────────────────────────────

class BLSTM:
    """
    Bidirectional LSTM with dual output heads for train/generate consistency.

    Parameters
    ----------
    vocab_size  : int
    hidden_size : int   hidden units per direction  (default 96)
    aux_weight  : float weight on the forward-only auxiliary loss (default 0.5)
    """

    def __init__(self, vocab_size, hidden_size=96, aux_weight=0.5, seed=42):
        self.V   = vocab_size
        self.H   = hidden_size
        self.aux = aux_weight

        # Two independent LSTMs for forward and backward directions
        self.fwd = LSTMCell(vocab_size, hidden_size, seed=seed)
        self.bwd = LSTMCell(vocab_size, hidden_size, seed=seed + 1)

        V, H = vocab_size, hidden_size
        s_combined = np.sqrt(2.0 / (2 * H + V))
        s_fwd      = np.sqrt(2.0 / (H + V))

        # Combined output head [h_fwd || h_bwd] → V (used at training)
        self.Wc  = np.random.randn(V, 2 * H) * s_combined
        self.bc  = np.zeros((V, 1))
        self.mWc = np.zeros_like(self.Wc)
        self.mbc = np.zeros_like(self.bc)

        # Forward-only output head h_fwd → V (used at generation)
        self.Wf  = np.random.randn(V, H) * s_fwd
        self.bf  = np.zeros((V, 1))
        self.mWf = np.zeros_like(self.Wf)
        self.mbf = np.zeros_like(self.bf)

    def count_params(self):
        return (self.fwd.count_params() +
                self.bwd.count_params() +
                self.Wc.size + self.bc.size +
                self.Wf.size + self.bf.size)

    # ── training ─────────────────────────────────────────────────────────────

    def _adagrad(self, param, mem, grad, lr):
        """Helper for Adagrad update of individual parameters."""
        eps = 1e-8
        mem += grad ** 2
        param -= lr * grad / (np.sqrt(mem) + eps)

    def train(self, names, ch2id, epochs=100, lr=0.003, print_every=10):
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(names)

            for name in names:
                # Convert name to token sequence with start/end markers
                seq = ([ch2id['^']] +
                       [ch2id[c] for c in name if c in ch2id] +
                       [ch2id['$']])
                if len(seq) < 3:
                    continue
                T  = len(seq)
                xs = [one_hot(seq[t], self.V) for t in range(T)]

                # ── forward LSTM (left→right over input tokens 0..T-2) ──────
                fwd_hs, _, fwd_caches = self.fwd.forward_seq(xs[:-1])

                # ── backward LSTM (right→left over input tokens 1..T-1) ───────
                # Process reversed sequence, then reorder to align with forward
                bwd_hs, _, bwd_caches = self.bwd.forward_seq(list(reversed(xs[1:])))
                bwd_hs_ordered = list(reversed(bwd_hs))  # Align with forward time steps

                # ── output layer gradients ──────────────────────────────────────
                loss = 0.0
                dh_fwd = [np.zeros((self.H, 1)) for _ in range(T - 1)]
                dh_bwd = [np.zeros((self.H, 1)) for _ in range(T - 1)]
                dWc = np.zeros_like(self.Wc);  dbc = np.zeros_like(self.bc)
                dWf = np.zeros_like(self.Wf);  dbf = np.zeros_like(self.bf)

                for t in range(T - 1):
                    target = seq[t + 1]
                    hf = fwd_hs[t]
                    hb = bwd_hs_ordered[t]

                    # Combined head (main loss) - uses both directions
                    hcat = np.vstack([hf, hb])             # (2H, 1)
                    yc   = self.Wc @ hcat + self.bc
                    pc   = softmax(yc)
                    loss -= np.log(pc[target, 0] + 1e-12)
                    dyc  = pc.copy();  dyc[target] -= 1.0
                    dWc += dyc @ hcat.T
                    dbc += dyc
                    dhcat      = self.Wc.T @ dyc            # (2H, 1)
                    dh_fwd[t] += dhcat[:self.H]             # Forward portion
                    dh_bwd[t] += dhcat[self.H:]             # Backward portion

                    # Forward-only head (auxiliary loss) - ensures forward LSTM learns independently
                    yf  = self.Wf @ hf + self.bf
                    pf  = softmax(yf)
                    loss -= self.aux * np.log(pf[target, 0] + 1e-12)
                    dyf  = pf.copy();  dyf[target] -= 1.0;  dyf *= self.aux
                    dWf += dyf @ hf.T
                    dbf += dyf
                    dh_fwd[t] += self.Wf.T @ dyf

                epoch_loss += loss

                # Update output heads with gradient clipping
                np.clip(dWc, -5, 5, out=dWc);  np.clip(dbc, -5, 5, out=dbc)
                np.clip(dWf, -5, 5, out=dWf);  np.clip(dbf, -5, 5, out=dbf)
                self._adagrad(self.Wc,  self.mWc,  dWc,  lr)
                self._adagrad(self.bc,  self.mbc,  dbc,  lr)
                self._adagrad(self.Wf,  self.mWf,  dWf,  lr)
                self._adagrad(self.bf,  self.mbf,  dbf,  lr)

                # BPTT through LSTMs
                dc_zeros = [np.zeros((self.H, 1)) for _ in range(T - 1)]

                # Forward LSTM: backpropagate through time
                gf = self.fwd.backward_seq(fwd_caches, dh_fwd, dc_zeros)
                self.fwd.adagrad_step(*gf, lr)

                # Backward LSTM: reverse gradients to match original time order
                dh_bwd_rev = list(reversed(dh_bwd))
                gb = self.bwd.backward_seq(bwd_caches, dh_bwd_rev, dc_zeros)
                self.bwd.adagrad_step(*gb, lr)

            avg = epoch_loss / max(len(names), 1)
            losses.append(avg)
            if (epoch + 1) % print_every == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

        return losses

    # ── generation ───────────────────────────────────────────────────────────

    def generate(self, ch2id, id2ch, n=200, max_len=20, temperature=0.85, seed=None):
        """
        Autoregressive generation using the FORWARD LSTM + forward-only head.
        No backward LSTM involved — fully consistent with training.
        This is why we trained the auxiliary head: to avoid distribution mismatch.
        """
        if seed is not None:
            np.random.seed(seed)

        generated = []
        for _ in range(n):
            h   = np.zeros((self.H, 1))
            c   = np.zeros((self.H, 1))
            idx = ch2id['^']
            chars = []

            for _ in range(max_len):
                x      = one_hot(idx, self.V)
                h, c, _ = self.fwd.step(x, h, c)
                y      = self.Wf @ h + self.bf          # Forward-only head (not combined)
                y      = y / temperature                # Temperature scaling
                p      = softmax(y).ravel()
                idx    = np.random.choice(self.V, p=p)
                if idx == ch2id['$']:
                    break
                chars.append(id2ch[idx])

            if len(chars) >= 2:
                generated.append(''.join(chars).capitalize())

        return generated


# ─────────────────────── main ────────────────────────────────────────────────

if __name__ == '__main__':
    DATA = os.path.join(os.path.dirname(__file__), 'training_names.txt')
    names = load_names(DATA)
    ch2id, id2ch = build_vocab(names)

    print(f"Vocab: {len(ch2id)} chars  |  Training names: {len(names)}")

    model = BLSTM(vocab_size=len(ch2id), hidden_size=96, aux_weight=0.5)
    print(f"Trainable parameters: {model.count_params():,}")
    print("Architecture : Fwd-LSTM(96) + Bwd-LSTM(96) → Combined-head(192→V) [train]")
    print("             : Fwd-LSTM(96) → Fwd-only-head(96→V)                  [generate]")
    print("Hyperparams  : hidden=96/dir, lr=0.003, aux_weight=0.5, clip=5.0, temp=0.85\n")

    losses = model.train(names, ch2id, epochs=100, lr=0.003, print_every=10)

    gen = model.generate(ch2id, id2ch, n=201, temperature=0.85, seed=42)

    out = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'blstm_names.txt')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w').write('\n'.join(gen))

    print(f"\nGenerated {len(gen)} names  →  {out}")
    print("Sample:", gen[:20])