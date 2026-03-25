"""
TASK-1: Bidirectional LSTM (BLSTM) for Character-Level Name Generation
Implemented from scratch using NumPy (no deep learning frameworks).

Architecture
------------
Training:
  - Forward LSTM  (left->right) over the input prefix
  - Backward LSTM (right->left) over the reversed prefix
  - Concatenated hidden states [h_fwd || h_bwd] fed to a COMBINED output head
  - Forward hidden states h_fwd also fed to a FORWARD-ONLY output head
    (trained with auxiliary weight 0.5 so it learns a standalone LM)

Generation (autoregressive -- cannot use backward direction):
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


# ─────────────────────── Utility functions ───────────────────────────────────

def load_names(path):
    """Read names from file, convert to lowercase, strip whitespace."""
    with open(path) as f:
        return [line.strip().lower() for line in f if line.strip()]

def build_vocab(names):
    """Create character mappings with start (^) and end ($) tokens."""
    chars = sorted(set(''.join(names)))
    chars = ['^'] + chars + ['$']
    ch2id = {c: i for i, c in enumerate(chars)}
    id2ch = {i: c for c, i in ch2id.items()}
    return ch2id, id2ch

def sigmoid(x):
    """Numerically stable sigmoid: handles large positive/negative values."""
    # Use two branches to avoid overflow in both exp(-x) and exp(x)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()

def one_hot(idx, size):
    """Create a one-hot column vector for the given index."""
    x = np.zeros((size, 1))
    x[idx] = 1.0
    return x


# ─────────────────────── LSTM cell ───────────────────────────────────────────

class LSTMCell:
    """Single-direction LSTM layer with packed gate matrices.

    All four gates [forget, input, candidate, output] are packed into a single
    weight matrix for efficiency: Wx has shape (4H, I) and Wh has shape (4H, H).
    """

    def __init__(self, input_size, hidden_size, seed=0):
        np.random.seed(seed)
        H, I = hidden_size, input_size
        s = np.sqrt(2.0 / (I + H))   # Xavier initialisation scale

        # Packed weight matrices: each row block of size H corresponds to one gate
        # Layout: [forget | input | candidate | output] stacked vertically
        self.Wx = np.random.randn(4 * H, I) * s   # input -> all gates
        self.Wh = np.random.randn(4 * H, H) * s   # hidden -> all gates
        self.b  = np.zeros((4 * H, 1))             # combined gate biases
        self.H  = H

        # Adagrad memory buffers (accumulate squared gradients per-parameter)
        self.mWx = np.zeros_like(self.Wx)
        self.mWh = np.zeros_like(self.Wh)
        self.mb  = np.zeros_like(self.b)

    def count_params(self):
        """Return total number of trainable scalar parameters."""
        return self.Wx.size + self.Wh.size + self.b.size

    def step(self, x, h_prev, c_prev):
        """Single LSTM step: compute new hidden and cell states.

        Parameters
        ----------
        x      : np.ndarray (I, 1)   current input vector
        h_prev : np.ndarray (H, 1)   previous hidden state
        c_prev : np.ndarray (H, 1)   previous cell state

        Returns
        -------
        h     : new hidden state (H, 1)
        c     : new cell state   (H, 1)
        cache : tuple of all values needed for the backward pass
        """
        H = self.H
        # Compute all gate pre-activations in one matmul for efficiency
        z = self.Wx @ x + self.Wh @ h_prev + self.b   # (4H, 1)

        # Split packed vector into individual gates and apply activations
        f = sigmoid(z[:H])       # Forget gate:    how much of c_prev to keep
        i = sigmoid(z[H:2*H])    # Input gate:     how much of candidate to write
        g = np.tanh(z[2*H:3*H])  # Candidate cell: proposed new content
        o = sigmoid(z[3*H:])     # Output gate:    how much of cell to expose

        # Update cell state: forget old + write new
        c = f * c_prev + i * g
        # New hidden state: gated output of the cell
        h = o * np.tanh(c)

        # Cache all intermediate values for BPTT
        return h, c, (x, h_prev, c_prev, f, i, g, o, c)

    def forward_seq(self, xs):
        """Process an entire sequence through the LSTM, step by step.

        Parameters
        ----------
        xs : list[np.ndarray]   sequence of input vectors

        Returns
        -------
        hs     : list of hidden states
        cs     : list of cell states
        caches : list of per-step caches for backward pass
        """
        H  = self.H
        h  = np.zeros((H, 1))   # Zero-initialise hidden state
        c  = np.zeros((H, 1))   # Zero-initialise cell state
        hs, cs, caches = [], [], []
        for x in xs:
            h, c, cache = self.step(x, h, c)
            hs.append(h)
            cs.append(c)
            caches.append(cache)
        return hs, cs, caches

    def backward_seq(self, caches, dh_list, dc_list, clip=5.0):
        """BPTT over the full sequence for one direction.

        Parameters
        ----------
        caches   : list[tuple]        per-step caches from forward_seq()
        dh_list  : list[np.ndarray]   gradient into h at each step from upper layers
        dc_list  : list[np.ndarray]   gradient into c at each step (usually zeros)
        clip     : float              gradient clipping threshold

        Returns
        -------
        dWx, dWh, db : gradients for the three parameter tensors
        """
        H   = self.H
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db  = np.zeros_like(self.b)
        dh_next = np.zeros((H, 1))   # Gradient arriving from the next timestep
        dc_next = np.zeros((H, 1))   # Cell-state gradient from the next timestep

        for t in reversed(range(len(caches))):
            x, h_prev, c_prev, f, i, g, o, c = caches[t]

            # Combine gradient from upper layer and from the future timestep
            dh = dh_list[t] + dh_next
            dc = dc_list[t] + dc_next

            # Gradient through the output gate and tanh(c)
            do = dh * np.tanh(c)
            dc = dc + dh * o * (1.0 - np.tanh(c) ** 2)   # Chain rule through tanh

            # Gradients into the cell update: dc -> df, di, dg
            df = dc * c_prev
            di = dc * g
            dg = dc * i

            # Pre-activation gradients via gate-specific derivative
            # sigmoid'(x) = sig(x) * (1 - sig(x)),  tanh'(x) = 1 - tanh^2(x)
            dz = np.vstack([
                df * f * (1.0 - f),      # forget gate pre-activation gradient
                di * i * (1.0 - i),      # input gate pre-activation gradient
                dg * (1.0 - g ** 2),     # candidate cell pre-activation gradient
                do * o * (1.0 - o),      # output gate pre-activation gradient
            ])

            # Accumulate weight gradients
            dWx     += dz @ x.T
            dWh     += dz @ h_prev.T
            db      += dz
            # Carry gradients back to previous timestep
            dh_next  = self.Wh.T @ dz
            dc_next  = dc * f   # Forget gate gates the cell gradient backward

        # Clip to prevent exploding gradients
        np.clip(dWx, -clip, clip, out=dWx)
        np.clip(dWh, -clip, clip, out=dWh)
        np.clip(db,  -clip, clip, out=db)
        return dWx, dWh, db

    def adagrad_step(self, dWx, dWh, db, lr):
        """Adagrad update with parameter-specific adaptive learning rates."""
        eps = 1e-8
        # Accumulate squared gradients and apply adaptive step
        self.mWx += dWx ** 2;  self.Wx -= lr * dWx / (np.sqrt(self.mWx) + eps)
        self.mWh += dWh ** 2;  self.Wh -= lr * dWh / (np.sqrt(self.mWh) + eps)
        self.mb  += db  ** 2;  self.b  -= lr * db  / (np.sqrt(self.mb)  + eps)


# ─────────────────────── BLSTM model ─────────────────────────────────────────

class BLSTM:
    """Bidirectional LSTM with dual output heads for train/generate consistency.

    Training uses BOTH forward and backward LSTM outputs (combined head).
    Generation uses ONLY the forward LSTM (forward-only head) because the
    backward direction is not available in an autoregressive left-to-right setting.
    The forward-only head is trained with an auxiliary loss during training to
    ensure it learns a strong standalone language model.

    Parameters
    ----------
    vocab_size  : int
    hidden_size : int   hidden units per direction  (default 96)
    aux_weight  : float weight on the forward-only auxiliary loss (default 0.5)
    """

    def __init__(self, vocab_size, hidden_size=96, aux_weight=0.5, seed=42):
        self.V   = vocab_size
        self.H   = hidden_size
        self.aux = aux_weight   # Scales the auxiliary loss term

        # Two independent LSTMs: one left-to-right, one right-to-left
        self.fwd = LSTMCell(vocab_size, hidden_size, seed=seed)
        self.bwd = LSTMCell(vocab_size, hidden_size, seed=seed + 1)

        V, H = vocab_size, hidden_size
        s_combined = np.sqrt(2.0 / (2 * H + V))  # Xavier for 2H -> V
        s_fwd      = np.sqrt(2.0 / (H + V))       # Xavier for H -> V

        # Combined head: takes concatenated [h_fwd || h_bwd] as input (used at training)
        self.Wc  = np.random.randn(V, 2 * H) * s_combined
        self.bc  = np.zeros((V, 1))
        self.mWc = np.zeros_like(self.Wc)   # Adagrad accumulator
        self.mbc = np.zeros_like(self.bc)

        # Forward-only head: takes only h_fwd as input (used at generation)
        self.Wf  = np.random.randn(V, H) * s_fwd
        self.bf  = np.zeros((V, 1))
        self.mWf = np.zeros_like(self.Wf)   # Adagrad accumulator
        self.mbf = np.zeros_like(self.bf)

    def count_params(self):
        """Return total number of trainable scalar parameters across all components."""
        return (self.fwd.count_params() +
                self.bwd.count_params() +
                self.Wc.size + self.bc.size +
                self.Wf.size + self.bf.size)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _adagrad(self, param, mem, grad, lr):
        """Adagrad in-place update for a single parameter tensor."""
        eps = 1e-8
        mem += grad ** 2
        param -= lr * grad / (np.sqrt(mem) + eps)

    # ── Training ─────────────────────────────────────────────────────────

    def train(self, names, ch2id, epochs=100, lr=0.003, print_every=10):
        """Train both LSTMs and both output heads over multiple epochs.

        For each name:
          1. Run forward LSTM left-to-right over tokens 0..T-2.
          2. Run backward LSTM right-to-left over tokens 1..T-1 (reversed).
          3. At each position t, compute:
               - Combined loss: cross-entropy from [h_fwd[t] || h_bwd[t]]
               - Auxiliary loss: cross-entropy from h_fwd[t] alone (weighted by aux)
          4. Backpropagate and update with Adagrad.

        Returns
        -------
        losses : list[float]   per-epoch average combined + auxiliary loss
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(names)   # Randomise order each epoch

            for name in names:
                # Build full token sequence: [start, c1, c2, ..., end]
                seq = ([ch2id['^']] +
                       [ch2id[c] for c in name if c in ch2id] +
                       [ch2id['$']])
                if len(seq) < 3:
                    continue   # Skip names that are too short
                T  = len(seq)
                xs = [one_hot(seq[t], self.V) for t in range(T)]   # Pre-compute one-hots

                # ── Forward LSTM: process tokens x_0 .. x_{T-2} left-to-right ──
                fwd_hs, _, fwd_caches = self.fwd.forward_seq(xs[:-1])

                # ── Backward LSTM: process tokens x_1 .. x_{T-1} right-to-left ──
                # We reverse the slice so the LSTM sees the sequence right-to-left
                bwd_hs, _, bwd_caches = self.bwd.forward_seq(list(reversed(xs[1:])))
                # Reverse the outputs back to align with forward time steps
                bwd_hs_ordered = list(reversed(bwd_hs))

                # ── Output layer: compute gradients for each timestep ──
                loss = 0.0
                # Gradient buffers for the LSTMs: filled in the output-layer loop
                dh_fwd = [np.zeros((self.H, 1)) for _ in range(T - 1)]
                dh_bwd = [np.zeros((self.H, 1)) for _ in range(T - 1)]
                dWc = np.zeros_like(self.Wc);  dbc = np.zeros_like(self.bc)
                dWf = np.zeros_like(self.Wf);  dbf = np.zeros_like(self.bf)

                for t in range(T - 1):
                    target = seq[t + 1]   # Character to predict at this step
                    hf = fwd_hs[t]         # Forward hidden state
                    hb = bwd_hs_ordered[t] # Backward hidden state (aligned)

                    # ── Combined head (main loss) ──────────────────────
                    hcat = np.vstack([hf, hb])             # Concatenate both directions: (2H, 1)
                    yc   = self.Wc @ hcat + self.bc
                    pc   = softmax(yc)
                    loss -= np.log(pc[target, 0] + 1e-12)  # Negative log-likelihood
                    dyc  = pc.copy();  dyc[target] -= 1.0  # dL/dy = softmax - one_hot
                    dWc += dyc @ hcat.T
                    dbc += dyc
                    dhcat      = self.Wc.T @ dyc            # Gradient into concatenated hidden
                    dh_fwd[t] += dhcat[:self.H]             # Distribute to forward LSTM
                    dh_bwd[t] += dhcat[self.H:]             # Distribute to backward LSTM

                    # ── Forward-only head (auxiliary loss) ────────────────
                    # This head trains the forward LSTM to be a good standalone LM,
                    # which is required for autoregressive generation at inference time
                    yf  = self.Wf @ hf + self.bf
                    pf  = softmax(yf)
                    loss -= self.aux * np.log(pf[target, 0] + 1e-12)   # Scaled auxiliary loss
                    dyf  = pf.copy();  dyf[target] -= 1.0;  dyf *= self.aux  # Scale gradient
                    dWf += dyf @ hf.T
                    dbf += dyf
                    dh_fwd[t] += self.Wf.T @ dyf   # This gradient only flows to forward LSTM

                epoch_loss += loss

                # ── Update output heads (with gradient clipping) ──────────
                np.clip(dWc, -5, 5, out=dWc);  np.clip(dbc, -5, 5, out=dbc)
                np.clip(dWf, -5, 5, out=dWf);  np.clip(dbf, -5, 5, out=dbf)
                self._adagrad(self.Wc,  self.mWc,  dWc,  lr)
                self._adagrad(self.bc,  self.mbc,  dbc,  lr)
                self._adagrad(self.Wf,  self.mWf,  dWf,  lr)
                self._adagrad(self.bf,  self.mbf,  dbf,  lr)

                # ── BPTT through both LSTMs ───────────────────────────────
                dc_zeros = [np.zeros((self.H, 1)) for _ in range(T - 1)]

                # Backprop through the forward LSTM (cell gradients are zero from above)
                gf = self.fwd.backward_seq(fwd_caches, dh_fwd, dc_zeros)
                self.fwd.adagrad_step(*gf, lr)

                # Backprop through the backward LSTM:
                # dh_bwd was assembled in forward-time order but the backward LSTM
                # processed the reversed sequence, so we must reverse the gradient list
                dh_bwd_rev = list(reversed(dh_bwd))
                gb = self.bwd.backward_seq(bwd_caches, dh_bwd_rev, dc_zeros)
                self.bwd.adagrad_step(*gb, lr)

            avg = epoch_loss / max(len(names), 1)
            losses.append(avg)
            if (epoch + 1) % print_every == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

        return losses

    # ── Generation ───────────────────────────────────────────────────────

    def generate(self, ch2id, id2ch, n=200, max_len=20, temperature=0.85, seed=None):
        """Autoregressive generation using the FORWARD LSTM + forward-only head.

        The backward LSTM is not used here because at generation time we do not
        yet know the right context. The forward-only head was specifically trained
        to produce good output from forward hidden states alone, avoiding the
        train/test distribution mismatch that would arise from using the combined head.
        """
        if seed is not None:
            np.random.seed(seed)

        generated = []
        for _ in range(n):
            # Zero-initialise LSTM state for each new name
            h   = np.zeros((self.H, 1))
            c   = np.zeros((self.H, 1))
            idx = ch2id['^']   # Start with the start token
            chars = []

            for _ in range(max_len):
                x      = one_hot(idx, self.V)
                h, c, _ = self.fwd.step(x, h, c)        # One forward LSTM step
                y      = self.Wf @ h + self.bf           # Forward-only head logits
                y      = y / temperature                  # Temperature scaling
                p      = softmax(y).ravel()
                idx    = np.random.choice(self.V, p=p)   # Sample next character
                if idx == ch2id['$']:
                    break   # Stop at end token
                chars.append(id2ch[idx])

            # Keep names of length >= 2 and capitalise
            if len(chars) >= 2:
                generated.append(''.join(chars).capitalize())

        return generated


# ─────────────────────── Main ────────────────────────────────────────────────

if __name__ == '__main__':
    DATA = os.path.join(os.path.dirname(__file__), 'training_names.txt')
    names = load_names(DATA)
    ch2id, id2ch = build_vocab(names)

    print(f"Vocab: {len(ch2id)} chars  |  Training names: {len(names)}")

    # Initialise and describe the model
    model = BLSTM(vocab_size=len(ch2id), hidden_size=96, aux_weight=0.5)
    print(f"Trainable parameters: {model.count_params():,}")
    print("Architecture : Fwd-LSTM(96) + Bwd-LSTM(96) -> Combined-head(192->V) [train]")
    print("             : Fwd-LSTM(96) -> Fwd-only-head(96->V)                  [generate]")
    print("Hyperparams  : hidden=96/dir, lr=0.003, aux_weight=0.5, clip=5.0, temp=0.85\n")

    # Train and collect per-epoch losses
    losses = model.train(names, ch2id, epochs=100, lr=0.003, print_every=10)

    # Generate 201 names (odd count avoids edge cases in some metrics)
    gen = model.generate(ch2id, id2ch, n=201, temperature=0.85, seed=42)

    # Save output
    out = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'blstm_names.txt')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w').write('\n'.join(gen))

    print(f"\nGenerated {len(gen)} names  ->  {out}")
    print("Sample:", gen[:20])
