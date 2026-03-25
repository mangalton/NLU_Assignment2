"""
TASK-1: RNN with Basic Attention Mechanism for Character-Level Name Generation
Implemented from scratch using NumPy (no deep learning frameworks).

Architecture:
  - Encoder RNN: reads the prefix x_0..x_{t-1}, produces hidden states
  - Additive (Bahdanau-style) attention: weighted context over encoder states
  - Decoder step: receives [x_t ; context] and predicts next character
  - Output linear layer with softmax

Hyperparameters
---------------
hidden_size  : 128  (encoder and decoder)
attn_size    : 64   (attention projection)
layers       : 1 encoder + 1 decoder
learning_rate: 0.005 (Adagrad), gradient clip = 5.0
"""

import numpy as np
import os


def load_names(path):
    with open(path) as f:
        return [l.strip().lower() for l in f if l.strip()]

def build_vocab(names):
    chars = sorted(set(''.join(names)))
    chars = ['^'] + chars + ['$']
    ch2id = {c: i for i, c in enumerate(chars)}
    id2ch = {i: c for c, i in ch2id.items()}
    return ch2id, id2ch

def one_hot(idx, size):
    """Create one-hot vector for given index."""
    x = np.zeros((size, 1))
    x[idx] = 1
    return x

def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()




class RNNAttention:
    """Single-layer Encoder-Decoder RNN with additive (Bahdanau) attention."""

    def __init__(self, vocab_size, hidden_size=128, attn_size=64, seed=42):
        np.random.seed(seed)
        V, H, A = vocab_size, hidden_size, attn_size
        self.V, self.H, self.A = V, H, A

        # Xavier-style initialization for stable gradients
        def g(r, c):
            return np.random.randn(r, c) * np.sqrt(2.0 / (r + c))

        # Encoder parameters
        self.Wxh_e = g(H, V);   self.Whh_e = g(H, H);   self.bh_e  = np.zeros((H,1))
        # Decoder parameters (input dimension V+H due to concatenated context)
        self.Wxh_d = g(H, V+H); self.Whh_d = g(H, H);   self.bh_d  = np.zeros((H,1))
        # Attention parameters: additive (Bahdanau) attention
        self.Wa    = g(A, H);   self.Ua    = g(A, H);   self.ba    = np.zeros((A,1))
        self.va    = g(1, A)    # Attention scoring vector
        # Output layer
        self.Why   = g(V, H);   self.by    = np.zeros((V,1))

        # Adagrad accumulators for all parameters
        self._pnames = ['Wxh_e','Whh_e','bh_e','Wxh_d','Whh_d','bh_d',
                        'Wa','Ua','ba','va','Why','by']
        for n in self._pnames:
            setattr(self, 'm_'+n, np.zeros_like(getattr(self, n)))

    def count_params(self):
        return sum(getattr(self, n).size for n in self._pnames)

    def _encode(self, xs):
        """Forward pass through encoder RNN. Returns list of hidden states."""
        h = np.zeros((self.H, 1))
        hs = []
        for x in xs:
            h = np.tanh(self.Wxh_e @ x + self.Whh_e @ h + self.bh_e)
            hs.append(h.copy())
        return hs

    def _attend(self, s, enc_hs):
        """
        Bahdanau attention mechanism.
        Computes context vector from encoder states using current decoder state.
        """
        if not enc_hs:
            return np.zeros((self.H, 1)), np.array([1.0])
        
        # Score each encoder state: va @ tanh(Wa·s + Ua·h + ba)
        Wa_s = self.Wa @ s
        scores = np.array([(self.va @ np.tanh(Wa_s + self.Ua @ h + self.ba)).ravel()[0]
                           for h in enc_hs])
        scores -= scores.max()  # Numerical stability for softmax
        alpha = np.exp(scores); alpha /= alpha.sum()
        
        # Weighted sum of encoder states
        context = sum(a * h for a, h in zip(alpha, enc_hs))
        return context, alpha

    def _forward(self, seq):
        """
        Forward pass for a single sequence.
        Returns: loss, caches (for BPTT), encoder states, inputs
        """
        T   = len(seq)
        xs  = [one_hot(seq[t], self.V) for t in range(T)]
        enc_hs = self._encode(xs[:-1])  # Encode all but last (no target after last)
        s   = np.zeros((self.H, 1))
        loss, caches = 0.0, []
        
        for t in range(T - 1):
            # Attend only to positions up to current time step (causal attention)
            avail = enc_hs[:t+1] if enc_hs else []
            context, alpha = self._attend(s, avail)
            
            # Decoder input: concatenate current input with context
            dec_in  = np.vstack([xs[t], context])
            s_new   = np.tanh(self.Wxh_d @ dec_in + self.Whh_d @ s + self.bh_d)
            y       = self.Why @ s_new + self.by
            p       = softmax(y)
            target  = seq[t+1]
            loss   -= np.log(p[target, 0] + 1e-12)
            
            # Cache everything for backward pass
            caches.append({'xs_t': xs[t], 'context': context, 'alpha': alpha,
                           'avail': avail, 's_prev': s, 's_new': s_new,
                           'dec_in': dec_in, 'p': p, 'target': target})
            s = s_new
        return loss, caches, enc_hs, xs

    def _backward(self, caches, enc_hs, xs, clip=5.0):
        """Backprop through time with gradient clipping."""
        H, V = self.H, self.V
        grads   = {n: np.zeros_like(getattr(self, n)) for n in self._pnames}
        denc_hs = [np.zeros((H, 1)) for _ in enc_hs]  # Gradients from decoder to encoder
        ds_next = np.zeros((H, 1))

        # Reverse through time steps
        for t in reversed(range(len(caches))):
            c = caches[t]
            p, target, s_new, s_prev = c['p'], c['target'], c['s_new'], c['s_prev']
            dec_in, context = c['dec_in'], c['context']
            alpha, avail    = c['alpha'], c['avail']

            # Output layer gradient (softmax + cross-entropy)
            dy = p.copy(); dy[target] -= 1
            grads['Why'] += dy @ s_new.T
            grads['by']  += dy
            ds = self.Why.T @ dy + ds_next

            # Decoder hidden state gradient with tanh derivative
            ds_raw = (1 - s_new**2) * ds
            grads['Wxh_d'] += ds_raw @ dec_in.T
            grads['Whh_d'] += ds_raw @ s_prev.T
            grads['bh_d']  += ds_raw
            ds_next = self.Whh_d.T @ ds_raw

            # Gradient through concatenated decoder input
            d_dec_in  = self.Wxh_d.T @ ds_raw
            d_context = d_dec_in[V:]  # Context portion of the gradient

            # Attention gradients (complex due to softmax over scores)
            if avail:
                K    = len(avail)
                Wa_s = self.Wa @ s_prev
                pre  = [np.tanh(Wa_s + self.Ua @ avail[k] + self.ba) for k in range(K)]
                dots = np.array([(d_context.T @ avail[k]).ravel()[0] for k in range(K)])
                dot_sum = float(alpha @ dots)

                for k in range(K):
                    denc_hs[k] += alpha[k] * d_context
                    # Softmax derivative chain rule
                    d_score_k   = alpha[k] * (dots[k] - dot_sum)
                    d_pre_k     = self.va.T * d_score_k
                    d_tanh_k    = (1 - pre[k]**2) * d_pre_k
                    grads['va'] += (pre[k].T * d_score_k)
                    grads['Wa'] += d_tanh_k @ s_prev.T
                    grads['Ua'] += d_tanh_k @ avail[k].T
                    grads['ba'] += d_tanh_k

        # Backward through encoder (reverse time)
        dh = np.zeros((H, 1))
        for t in reversed(range(len(enc_hs))):
            h_prev = enc_hs[t-1] if t > 0 else np.zeros((H, 1))
            dh    += denc_hs[t]
            dh_raw = (1 - enc_hs[t]**2) * dh  # tanh derivative
            grads['Wxh_e'] += dh_raw @ xs[t].T
            grads['Whh_e'] += dh_raw @ h_prev.T
            grads['bh_e']  += dh_raw
            dh = self.Whh_e.T @ dh_raw

        # Clip gradients to prevent explosion
        for g in grads.values():
            np.clip(g, -clip, clip, out=g)
        return grads

    def _adagrad(self, grads, lr=0.005):
        """Update parameters using Adagrad optimizer."""
        eps = 1e-8
        for n, grad in grads.items():
            mem = getattr(self, 'm_'+n)
            mem += grad**2
            getattr(self, n).__isub__(lr * grad / (np.sqrt(mem) + eps))

    def train(self, names, ch2id, epochs=100, lr=0.005, print_every=10):
        """Train the attention model over multiple epochs."""
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(names)
            for name in names:
                seq = ([ch2id['^']] +
                       [ch2id[c] for c in name if c in ch2id] +
                       [ch2id['$']])
                if len(seq) < 3:
                    continue
                loss, caches, enc_hs, xs = self._forward(seq)
                epoch_loss += loss
                grads = self._backward(caches, enc_hs, xs)
                self._adagrad(grads, lr)
            avg = epoch_loss / max(len(names), 1)
            losses.append(avg)
            if (epoch + 1) % print_every == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")
        return losses

    def generate(self, ch2id, id2ch, n=200, max_len=20, temperature=0.85, seed=None):
        """Generate names by sampling with attention."""
        if seed is not None:
            np.random.seed(seed)
        generated = []
        for _ in range(n):
            idx_seq = [ch2id['^']]
            s = np.zeros((self.H, 1))
            for _ in range(max_len):
                # Encode all seen characters
                enc_hs  = self._encode([one_hot(i, self.V) for i in idx_seq])
                context, _ = self._attend(s, enc_hs)
                x_t     = one_hot(idx_seq[-1], self.V)
                dec_in  = np.vstack([x_t, context])
                s       = np.tanh(self.Wxh_d @ dec_in + self.Whh_d @ s + self.bh_d)
                y       = self.Why @ s + self.by
                y       = y / temperature  # Temperature scaling for diversity
                p       = softmax(y).ravel()
                idx     = np.random.choice(self.V, p=p)
                if idx == ch2id['$']:
                    break
                idx_seq.append(idx)
            chars = [id2ch[i] for i in idx_seq[1:]]
            if len(chars) >= 2:
                generated.append(''.join(chars).capitalize())
        return generated


if __name__ == '__main__':
    DATA = os.path.join(os.path.dirname(__file__), 'training_names.txt')
    names = load_names(DATA)
    ch2id, id2ch = build_vocab(names)
    model = RNNAttention(vocab_size=len(ch2id), hidden_size=128, attn_size=64)
    print(f"Params: {model.count_params():,}")
    losses = model.train(names, ch2id, epochs=100, lr=0.005, print_every=10)
    gen = model.generate(ch2id, id2ch, n=200, seed=7)
    out = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'rnn_attention_names.txt')
    open(out, 'w').write('\n'.join(gen))
    print("Sample:", gen[:15])