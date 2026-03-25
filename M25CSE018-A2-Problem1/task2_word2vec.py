"""
TASK 2: WORD2VEC MODEL TRAINING
Implements both CBOW and Skip-gram with Negative Sampling from scratch using NumPy.
"""

import numpy as np
import random
import collections
import math
import pickle
import time

# ─── VOCABULARY ────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, min_count=2):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}
        self.vocab_size = 0

    def build(self, sentences):
        freq = collections.Counter()
        for sent in sentences:
            freq.update(sent)
        # Filter by min_count
        self.word_counts = {w: c for w, c in freq.items() if c >= self.min_count}
        # Special pad token
        vocab_words = sorted(self.word_counts.keys())
        for i, w in enumerate(vocab_words):
            self.word2idx[w] = i
            self.idx2word[i] = w
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built: {self.vocab_size} words (min_count={self.min_count})")
        return self

    def encode_sentences(self, sentences):
        """Convert sentences to lists of indices, filtering unknown words"""
        encoded = []
        for sent in sentences:
            indices = [self.word2idx[w] for w in sent if w in self.word2idx]
            if len(indices) >= 2:
                encoded.append(indices)
        return encoded

    def get_negative_sampling_table(self, table_size=1_000_000):
        """Build unigram table for negative sampling (raised to 3/4 power)"""
        counts = np.array([self.word_counts.get(self.idx2word[i], 0)
                           for i in range(self.vocab_size)], dtype=np.float64)
        counts = counts ** 0.75
        counts /= counts.sum()
        table = np.zeros(table_size, dtype=np.int32)
        idx = 0
        cumulative = 0.0
        for i, prob in enumerate(counts):
            cumulative += prob
            while idx < table_size and idx / table_size < cumulative:
                table[idx] = i
                idx += 1
        return table


# ─── WORD2VEC BASE ─────────────────────────────────────────────────────────────

class Word2Vec:
    """
    Base Word2Vec implementation with Negative Sampling.
    Supports both CBOW and Skip-gram architectures.
    """
    def __init__(self, vocab, embedding_dim=100, window=5, neg_samples=5,
                 learning_rate=0.025, architecture='skipgram'):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.window = window
        self.neg_samples = neg_samples
        self.lr = learning_rate
        self.architecture = architecture
        self.V = vocab.vocab_size

        # Initialize weight matrices
        # W_in: V × D  (input/center word embeddings)
        # W_out: V × D (output/context word embeddings)
        self.W_in  = (np.random.rand(self.V, embedding_dim) - 0.5) / embedding_dim
        self.W_out = np.zeros((self.V, embedding_dim))

        # Build negative sampling table
        self.neg_table = vocab.get_negative_sampling_table()

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _get_negatives(self, pos_indices, n):
        """Sample n negative word indices, avoiding positives"""
        negs = []
        pos_set = set(pos_indices)
        while len(negs) < n:
            idx = self.neg_table[random.randint(0, len(self.neg_table) - 1)]
            if idx not in pos_set:
                negs.append(idx)
        return negs

    def _skipgram_update(self, center_idx, context_idx, neg_indices):
        """Single skip-gram + negative sampling update. Returns loss."""
        center_vec = self.W_in[center_idx]      # (D,)
        loss = 0.0

        # Positive sample
        score_pos = self._sigmoid(np.dot(center_vec, self.W_out[context_idx]))
        loss += -math.log(score_pos + 1e-10)
        grad_center = (score_pos - 1.0) * self.W_out[context_idx]
        self.W_out[context_idx] -= self.lr * (score_pos - 1.0) * center_vec

        # Negative samples
        for neg_idx in neg_indices:
            score_neg = self._sigmoid(np.dot(center_vec, self.W_out[neg_idx]))
            loss += -math.log(1.0 - score_neg + 1e-10)
            grad_center += score_neg * self.W_out[neg_idx]
            self.W_out[neg_idx] -= self.lr * score_neg * center_vec

        self.W_in[center_idx] -= self.lr * grad_center
        return loss

    def _cbow_update(self, context_indices, center_idx, neg_indices):
        """Single CBOW + negative sampling update. Returns loss."""
        if not context_indices:
            return 0.0
        # Average context vectors
        context_vecs = self.W_in[context_indices]    # (C, D)
        h = context_vecs.mean(axis=0)                # (D,)
        loss = 0.0

        # Positive
        score_pos = self._sigmoid(np.dot(h, self.W_out[center_idx]))
        loss += -math.log(score_pos + 1e-10)
        grad_h = (score_pos - 1.0) * self.W_out[center_idx]
        self.W_out[center_idx] -= self.lr * (score_pos - 1.0) * h

        # Negatives
        for neg_idx in neg_indices:
            score_neg = self._sigmoid(np.dot(h, self.W_out[neg_idx]))
            loss += -math.log(1.0 - score_neg + 1e-10)
            grad_h += score_neg * self.W_out[neg_idx]
            self.W_out[neg_idx] -= self.lr * score_neg * h

        # Update context word embeddings
        grad_input = grad_h / len(context_indices)
        for ctx_idx in context_indices:
            self.W_in[ctx_idx] -= self.lr * grad_input

        return loss

    def train(self, encoded_sentences, epochs=10, verbose=True):
        """Train Word2Vec model"""
        total_pairs = sum(len(s) for s in encoded_sentences)
        start = time.time()

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            # Shuffle sentences each epoch
            shuffled = encoded_sentences[:]
            random.shuffle(shuffled)

            # Linear LR decay
            epoch_lr = max(self.lr * (1 - epoch / epochs), self.lr * 0.0001)
            self.lr = epoch_lr

            for sent in shuffled:
                for i, center in enumerate(sent):
                    # Dynamic window size (uniformly sample 1..window)
                    w = random.randint(1, self.window)
                    ctx_indices = [sent[j] for j in range(max(0, i - w),
                                                          min(len(sent), i + w + 1))
                                   if j != i]
                    if not ctx_indices:
                        continue

                    neg_indices = self._get_negatives([center] + ctx_indices,
                                                      self.neg_samples)

                    if self.architecture == 'skipgram':
                        for ctx_idx in ctx_indices:
                            loss = self._skipgram_update(center, ctx_idx, neg_indices)
                            total_loss += loss
                            count += 1
                    else:  # cbow
                        loss = self._cbow_update(ctx_indices, center, neg_indices)
                        total_loss += loss
                        count += 1

            avg_loss = total_loss / max(count, 1)
            elapsed = time.time() - start
            if verbose:
                print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"LR: {self.lr:.6f} | Time: {elapsed:.1f}s")

        print(f"Training complete in {time.time()-start:.1f}s")

    def get_vector(self, word):
        """Return embedding for a word"""
        idx = self.vocab.word2idx.get(word)
        if idx is None:
            return None
        return self.W_in[idx]

    def most_similar(self, word, topn=5):
        """Return top-n most similar words by cosine similarity"""
        vec = self.get_vector(word)
        if vec is None:
            return []

        # Normalize query
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec_norm = vec / norm

        # Compute similarities with all vocab vectors
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed = self.W_in / norms
        sims = normed @ vec_norm     # (V,)

        # Exclude the query word itself
        query_idx = self.vocab.word2idx[word]
        sims[query_idx] = -1.0

        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_indices]

    def analogy(self, word_a, word_b, word_c, topn=5):
        """
        Solve: word_a : word_b :: word_c : ?
        Using: vec(b) - vec(a) + vec(c)
        """
        va, vb, vc = self.get_vector(word_a), self.get_vector(word_b), self.get_vector(word_c)
        missing = [w for w, v in zip([word_a, word_b, word_c], [va, vb, vc]) if v is None]
        if missing:
            return [], f"Words not in vocabulary: {missing}"

        query = vb - va + vc
        norm = np.linalg.norm(query)
        if norm == 0:
            return [], "Zero vector"
        query /= norm

        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed = self.W_in / norms
        sims = normed @ query

        # Exclude input words
        for w in [word_a, word_b, word_c]:
            idx = self.vocab.word2idx.get(w)
            if idx is not None:
                sims[idx] = -1.0

        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_indices], "ok"

    def cosine_similarity(self, word1, word2):
        v1, v2 = self.get_vector(word1), self.get_vector(word2)
        if v1 is None or v2 is None:
            return None
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'W_in': self.W_in, 'W_out': self.W_out,
                         'architecture': self.architecture,
                         'embedding_dim': self.embedding_dim,
                         'window': self.window,
                         'neg_samples': self.neg_samples}, f)
        print(f"Model saved to {path}")


# ─── EXPERIMENT RUNNER ─────────────────────────────────────────────────────────

def run_experiments(sentences, vocab):
    """Train multiple models with varying hyperparameters"""
    encoded = vocab.encode_sentences(sentences)
    print(f"\nEncoded {len(encoded)} sentences for training.\n")

    configs = [
        {'arch': 'skipgram', 'dim': 100, 'window': 5, 'neg': 5,  'label': 'SG-100d-w5-n5'},
        {'arch': 'skipgram', 'dim': 50,  'window': 3, 'neg': 5,  'label': 'SG-50d-w3-n5'},
        {'arch': 'skipgram', 'dim': 100, 'window': 7, 'neg': 10, 'label': 'SG-100d-w7-n10'},
        {'arch': 'cbow',     'dim': 100, 'window': 5, 'neg': 5,  'label': 'CBOW-100d-w5-n5'},
        {'arch': 'cbow',     'dim': 50,  'window': 3, 'neg': 5,  'label': 'CBOW-50d-w3-n5'},
        {'arch': 'cbow',     'dim': 100, 'window': 7, 'neg': 10, 'label': 'CBOW-100d-w7-n10'},
    ]

    models = {}
    for cfg in configs:
        print(f"\n{'='*55}")
        print(f"Training: {cfg['label']}")
        print(f"  arch={cfg['arch']}, dim={cfg['dim']}, "
              f"window={cfg['window']}, neg_samples={cfg['neg']}")
        print(f"{'='*55}")
        model = Word2Vec(vocab,
                         embedding_dim=cfg['dim'],
                         window=cfg['window'],
                         neg_samples=cfg['neg'],
                         learning_rate=0.025,
                         architecture=cfg['arch'])
        model.train(encoded, epochs=15, verbose=True)
        model.save(f"./model_{cfg['label']}.pkl")
        models[cfg['label']] = model

    return models, configs


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'os.getcwd()')
    from task1_preprocessing import build_corpus, compute_stats
    from iitj_corpus_raw import get_corpus

    documents = get_corpus()
    sentences = build_corpus(documents)
    stats = compute_stats(sentences, documents)
    print(f"Corpus: {stats['num_sentences']} sentences, "
          f"{stats['total_tokens']} tokens, vocab {stats['vocab_size']}")

    vocab = Vocabulary(min_count=2)
    vocab.build(sentences)

    models, configs = run_experiments(sentences, vocab)
    print("\nAll models trained successfully!")