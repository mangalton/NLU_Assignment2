"""
TASK 2: WORD2VEC MODEL TRAINING
Implements both CBOW and Skip-gram with Negative Sampling from scratch using NumPy.

Architecture overview:
  - Vocabulary  : builds word ↔ index mappings and a negative-sampling table
  - Word2Vec    : base model supporting both architectures with SGD updates
  - run_experiments : convenience function to train a grid of hyperparameter configs
"""

import numpy as np
import random
import collections
import math
import pickle
import time


# ─── VOCABULARY ────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Manages the mapping between words and integer indices, and builds
    the unigram frequency table used for negative sampling.

    Attributes:
        min_count (int)   : Minimum frequency for a word to be included.
        word2idx  (dict)  : word (str) → index (int)
        idx2word  (dict)  : index (int) → word (str)
        word_counts (dict): word (str) → raw frequency count
        vocab_size (int)  : total number of words in the vocabulary
    """

    def __init__(self, min_count=2):
        """
        Args:
            min_count (int): Words appearing fewer than this many times
                             in the corpus are discarded. Default: 2.
        """
        self.min_count = min_count
        self.word2idx = {}     # word → integer index
        self.idx2word = {}     # integer index → word (for decoding)
        self.word_counts = {}  # word → raw corpus frequency
        self.vocab_size = 0    # number of unique words after filtering

    def build(self, sentences):
        """
        Count token frequencies across all sentences and create bidirectional
        word ↔ index mappings, filtering out rare words.

        Args:
            sentences (list of list of str): Tokenized sentences.

        Returns:
            self: Fluent interface — allows chained calls.
        """
        # Count every token occurrence across the entire corpus
        freq = collections.Counter()
        for sent in sentences:
            freq.update(sent)

        # Discard words below min_count threshold (reduces noise and memory)
        self.word_counts = {w: c for w, c in freq.items() if c >= self.min_count}

        # Sort alphabetically for deterministic index assignment
        vocab_words = sorted(self.word_counts.keys())
        for i, w in enumerate(vocab_words):
            self.word2idx[w] = i   # forward lookup: word → id
            self.idx2word[i] = w   # reverse lookup: id → word

        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built: {self.vocab_size} words (min_count={self.min_count})")
        return self

    def encode_sentences(self, sentences):
        """
        Convert a list of token sentences into lists of integer indices,
        skipping any token not in the vocabulary (OOV words).

        Args:
            sentences (list of list of str): Tokenized sentences.

        Returns:
            list of list of int: Index-encoded sentences.
                                 Sentences that become shorter than 2
                                 tokens after OOV filtering are dropped.
        """
        encoded = []
        for sent in sentences:
            # Map each known word to its index; silently skip unknowns
            indices = [self.word2idx[w] for w in sent if w in self.word2idx]
            # A sentence must have at least 2 tokens to form any (center, context) pair
            if len(indices) >= 2:
                encoded.append(indices)
        return encoded

    def get_negative_sampling_table(self, table_size=1_000_000):
        """
        Build the large unigram lookup table used for fast negative sampling.

        Each index in the table holds a word index proportional to the word's
        frequency raised to the 3/4 power (as used in the original Word2Vec paper).
        Raising to 0.75 smooths the distribution — it reduces the dominance of
        very frequent words and gives rare words a slightly higher chance of
        being sampled as negatives.

        Args:
            table_size (int): Number of entries in the table. Larger → smoother
                              distribution but more memory. Default: 1,000,000.

        Returns:
            np.ndarray of dtype int32: The negative sampling lookup table.
        """
        # Build a float array of raw frequencies in vocabulary index order
        counts = np.array(
            [self.word_counts.get(self.idx2word[i], 0) for i in range(self.vocab_size)],
            dtype=np.float64
        )

        # Apply the 0.75 smoothing exponent from the original Word2Vec paper
        counts = counts ** 0.75

        # Normalize to a proper probability distribution
        counts /= counts.sum()

        # Pre-allocate the table
        table = np.zeros(table_size, dtype=np.int32)

        # Fill the table: each word fills a contiguous segment proportional to its prob
        idx = 0
        cumulative = 0.0
        for i, prob in enumerate(counts):
            cumulative += prob
            # Advance the table pointer as long as the current position is
            # still within the segment for word i
            while idx < table_size and idx / table_size < cumulative:
                table[idx] = i
                idx += 1

        return table


# ─── WORD2VEC BASE ─────────────────────────────────────────────────────────────

class Word2Vec:
    """
    From-scratch NumPy implementation of Word2Vec with Negative Sampling.
    Supports both Skip-gram and CBOW architectures.

    The model maintains two weight matrices:
      W_in  (V × D): Input embeddings (center-word / context-word for CBOW).
                     These are the embeddings used for downstream tasks.
      W_out (V × D): Output embeddings (predicted word representations).
                     Initialized to zeros; updated only during training.

    Training uses Stochastic Gradient Descent with a linearly decaying
    learning rate and dynamic window size sampling.
    """

    def __init__(self, vocab, embedding_dim=100, window=5, neg_samples=5,
                 learning_rate=0.025, architecture='skipgram'):
        """
        Args:
            vocab          (Vocabulary): Pre-built vocabulary object.
            embedding_dim  (int)       : Dimensionality of word vectors (D). Default: 100.
            window         (int)       : Maximum context window radius. Default: 5.
            neg_samples    (int)       : Number of negative samples per positive pair. Default: 5.
            learning_rate  (float)     : Initial SGD learning rate. Default: 0.025.
            architecture   (str)       : 'skipgram' or 'cbow'. Default: 'skipgram'.
        """
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.window = window
        self.neg_samples = neg_samples
        self.lr = learning_rate
        self.architecture = architecture
        self.V = vocab.vocab_size  # Shorthand for vocabulary size

        # ── Weight Matrices ──
        # W_in: each row is the embedding vector for word i (input/center side)
        # Initialized with small random values centered around 0 (range −0.5/D … +0.5/D)
        self.W_in  = (np.random.rand(self.V, embedding_dim) - 0.5) / embedding_dim

        # W_out: output side embeddings, initialized to zero as per the original paper
        self.W_out = np.zeros((self.V, embedding_dim))

        # Build the negative-sampling lookup table (large unigram array)
        self.neg_table = vocab.get_negative_sampling_table()

    def _sigmoid(self, x):
        """
        Numerically stable sigmoid function.
        Clips input to [-10, 10] to prevent overflow in exp().

        Args:
            x (float or np.ndarray): Input value(s).

        Returns:
            float or np.ndarray: Sigmoid-transformed value(s) in (0, 1).
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _get_negatives(self, pos_indices, n):
        """
        Draw n negative word indices from the noise distribution,
        ensuring none of them are the actual positive (context) words.

        Args:
            pos_indices (list of int): Indices of the positive (true) words
                                       to exclude from negative samples.
            n           (int)        : Number of negative samples to generate.

        Returns:
            list of int: Negative word indices of length n.
        """
        negs = []
        pos_set = set(pos_indices)  # Fast O(1) exclusion check

        while len(negs) < n:
            # Pick a random position in the pre-built unigram table
            idx = self.neg_table[random.randint(0, len(self.neg_table) - 1)]
            # Reject if this word is one of the positive context words
            if idx not in pos_set:
                negs.append(idx)

        return negs

    def _skipgram_update(self, center_idx, context_idx, neg_indices):
        """
        Perform one Skip-gram parameter update with negative sampling.

        Skip-gram objective: Given the CENTER word, predict each CONTEXT word.
        The model is trained to:
          - Maximize  P(context | center)  → sigmoid(center · context_out) → 1
          - Minimize  P(negative | center) → sigmoid(center · neg_out)     → 0

        Gradients are computed analytically from the log-sigmoid loss and
        applied immediately (online SGD).

        Args:
            center_idx  (int)       : Index of the center word.
            context_idx (int)       : Index of the positive context word.
            neg_indices (list of int): Indices of the negative sample words.

        Returns:
            float: The total loss for this (center, context, negatives) triplet.
        """
        center_vec = self.W_in[center_idx]   # Current embedding for center word (D,)
        loss = 0.0

        # ── Positive sample update ──
        # score_pos = σ(center · context_out)
        # Ideal score_pos = 1  →  gradient = (score_pos - 1)
        score_pos = self._sigmoid(np.dot(center_vec, self.W_out[context_idx]))
        loss += -math.log(score_pos + 1e-10)               # Binary cross-entropy loss

        # Gradient accumulator for the center word embedding
        grad_center = (score_pos - 1.0) * self.W_out[context_idx]

        # Update the output (context-side) embedding for the positive word
        self.W_out[context_idx] -= self.lr * (score_pos - 1.0) * center_vec

        # ── Negative sample updates ──
        for neg_idx in neg_indices:
            # score_neg = σ(center · neg_out)
            # Ideal score_neg = 0  →  gradient = score_neg
            score_neg = self._sigmoid(np.dot(center_vec, self.W_out[neg_idx]))
            loss += -math.log(1.0 - score_neg + 1e-10)      # Maximize 1 - P(neg)

            # Accumulate gradient contribution from this negative word
            grad_center += score_neg * self.W_out[neg_idx]

            # Update the output embedding for this negative word
            self.W_out[neg_idx] -= self.lr * score_neg * center_vec

        # Apply the accumulated gradient to the center word's input embedding
        self.W_in[center_idx] -= self.lr * grad_center

        return loss

    def _cbow_update(self, context_indices, center_idx, neg_indices):
        """
        Perform one CBOW parameter update with negative sampling.

        CBOW objective: Given the CONTEXT words (averaged), predict the CENTER word.
        The hidden layer h = mean of context word embeddings (W_in rows).
        The gradient flows back equally to all context word embeddings.

        Args:
            context_indices (list of int): Indices of the context (surrounding) words.
            center_idx      (int)        : Index of the target (center) word.
            neg_indices     (list of int): Indices of the negative sample words.

        Returns:
            float: Total loss for this (context, center, negatives) triplet.
        """
        if not context_indices:
            return 0.0  # Nothing to update without context

        # ── Compute the hidden layer: average of context vectors ──
        context_vecs = self.W_in[context_indices]   # Shape: (C, D)
        h = context_vecs.mean(axis=0)               # Shape: (D,) — the CBOW hidden state

        loss = 0.0

        # ── Positive sample ──
        score_pos = self._sigmoid(np.dot(h, self.W_out[center_idx]))
        loss += -math.log(score_pos + 1e-10)

        # Gradient w.r.t. hidden state h
        grad_h = (score_pos - 1.0) * self.W_out[center_idx]

        # Update output embedding for the true center word
        self.W_out[center_idx] -= self.lr * (score_pos - 1.0) * h

        # ── Negative samples ──
        for neg_idx in neg_indices:
            score_neg = self._sigmoid(np.dot(h, self.W_out[neg_idx]))
            loss += -math.log(1.0 - score_neg + 1e-10)

            # Accumulate gradient of h from this negative word
            grad_h += score_neg * self.W_out[neg_idx]

            # Update output embedding for this negative word
            self.W_out[neg_idx] -= self.lr * score_neg * h

        # ── Back-propagate gradient equally to all context word input embeddings ──
        # Divide by number of context words because h = mean(context_vecs)
        grad_input = grad_h / len(context_indices)
        for ctx_idx in context_indices:
            self.W_in[ctx_idx] -= self.lr * grad_input

        return loss

    def train(self, encoded_sentences, epochs=10, verbose=True):
        """
        Train the Word2Vec model for a given number of epochs.

        Training procedure per epoch:
          1. Shuffle sentences for better SGD convergence.
          2. Linearly decay the learning rate towards lr * 0.0001.
          3. For each word in each sentence:
             a. Sample a random window size (dynamic windowing).
             b. Collect context word indices within the window.
             c. Sample negative words.
             d. Perform the appropriate update (skip-gram or CBOW).

        Args:
            encoded_sentences (list of list of int): Index-encoded sentences.
            epochs (int): Number of full passes over the training data. Default: 10.
            verbose (bool): Print per-epoch loss and timing. Default: True.
        """
        start = time.time()

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0

            # Shuffle sentences to avoid order bias during SGD
            shuffled = encoded_sentences[:]
            random.shuffle(shuffled)

            # Linear LR decay: starts at initial lr, decays to 0.01% of it
            epoch_lr = max(self.lr * (1 - epoch / epochs), self.lr * 0.0001)
            self.lr = epoch_lr

            for sent in shuffled:
                for i, center in enumerate(sent):
                    # Dynamic window: sample a random window radius 1..window
                    # This naturally down-weights distant context words on average
                    w = random.randint(1, self.window)

                    # Collect context word indices within the sampled window,
                    # excluding the center word position itself
                    ctx_indices = [sent[j] for j in range(max(0, i - w),
                                                           min(len(sent), i + w + 1))
                                   if j != i]

                    if not ctx_indices:
                        continue  # Skip if no context words are available

                    # Sample negative words (excluding center and context words)
                    neg_indices = self._get_negatives(
                        [center] + ctx_indices, self.neg_samples
                    )

                    if self.architecture == 'skipgram':
                        # Skip-gram: one update per (center, context_word) pair
                        for ctx_idx in ctx_indices:
                            loss = self._skipgram_update(center, ctx_idx, neg_indices)
                            total_loss += loss
                            count += 1
                    else:
                        # CBOW: one update per (context_window → center_word) event
                        loss = self._cbow_update(ctx_indices, center, neg_indices)
                        total_loss += loss
                        count += 1

            # Average loss over all update steps in this epoch
            avg_loss = total_loss / max(count, 1)
            elapsed = time.time() - start

            if verbose:
                print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"LR: {self.lr:.6f} | Time: {elapsed:.1f}s")

        print(f"Training complete in {time.time()-start:.1f}s")

    def get_vector(self, word):
        """
        Return the trained input embedding vector for a word.

        Args:
            word (str): The query word.

        Returns:
            np.ndarray of shape (D,) if word is in vocabulary, else None.
        """
        idx = self.vocab.word2idx.get(word)
        if idx is None:
            return None  # Out-of-vocabulary word
        return self.W_in[idx]

    def most_similar(self, word, topn=5):
        """
        Find the most semantically similar words using cosine similarity.
        All W_in vectors are normalized once for efficiency (batch dot product).

        Args:
            word  (str): Query word.
            topn  (int): Number of nearest neighbours to return. Default: 5.

        Returns:
            list of (str, float): Top-n (word, cosine_similarity) pairs,
                                  sorted descending. Empty list if OOV.
        """
        vec = self.get_vector(word)
        if vec is None:
            return []

        # Normalize the query vector to unit length
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec_norm = vec / norm

        # Normalize all vocabulary vectors (add eps to avoid division by zero)
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed = self.W_in / norms  # Shape: (V, D)

        # Batch cosine similarity: dot each row of normed with the query vector
        sims = normed @ vec_norm   # Shape: (V,)

        # Exclude the query word from results (similarity to itself = 1)
        query_idx = self.vocab.word2idx[word]
        sims[query_idx] = -1.0

        # Return top-n indices sorted by descending similarity
        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_indices]

    def analogy(self, word_a, word_b, word_c, topn=5):
        """
        Solve a word analogy: word_a : word_b :: word_c : ?

        Uses the classic vector arithmetic:
            query = vec(word_b) - vec(word_a) + vec(word_c)
        Then finds vocabulary words closest to `query` by cosine similarity.

        Args:
            word_a (str): Source word of the analogy pair.
            word_b (str): Target word of the analogy pair.
            word_c (str): The query word to complete the analogy.
            topn   (int): Number of candidates to return. Default: 5.

        Returns:
            tuple: (results, message) where:
                results — list of (word, score) sorted by score descending.
                message — 'ok' on success, or an error description.
        """
        va = self.get_vector(word_a)
        vb = self.get_vector(word_b)
        vc = self.get_vector(word_c)

        # Identify any input words that are out-of-vocabulary
        missing = [w for w, v in zip([word_a, word_b, word_c], [va, vb, vc]) if v is None]
        if missing:
            return [], f"Words not in vocabulary: {missing}"

        # Compute the analogy query vector and normalize it
        query = vb - va + vc
        norm = np.linalg.norm(query)
        if norm == 0:
            return [], "Zero vector"
        query /= norm

        # Batch cosine similarity against all vocabulary vectors
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed = self.W_in / norms
        sims = normed @ query

        # Exclude the three input words from the result set
        for w in [word_a, word_b, word_c]:
            idx = self.vocab.word2idx.get(w)
            if idx is not None:
                sims[idx] = -1.0

        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_indices], "ok"

    def cosine_similarity(self, word1, word2):
        """
        Compute the cosine similarity between two words' embedding vectors.

        Args:
            word1 (str): First word.
            word2 (str): Second word.

        Returns:
            float: Cosine similarity in [-1, 1], or None if either word is OOV.
        """
        v1, v2 = self.get_vector(word1), self.get_vector(word2)
        if v1 is None or v2 is None:
            return None  # Cannot compute if either word is not in vocabulary
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0   # Zero vector has undefined direction → return 0
        return float(np.dot(v1, v2) / (n1 * n2))

    def save(self, path):
        """
        Serialize the model's weight matrices and hyperparameters to a pickle file.

        Args:
            path (str): Destination file path (e.g. './model_SG-100d.pkl').
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'W_in':          self.W_in,
                'W_out':         self.W_out,
                'architecture':  self.architecture,
                'embedding_dim': self.embedding_dim,
                'window':        self.window,
                'neg_samples':   self.neg_samples
            }, f)
        print(f"Model saved to {path}")


# ─── EXPERIMENT RUNNER ─────────────────────────────────────────────────────────

def run_experiments(sentences, vocab):
    """
    Train a grid of Word2Vec models with different hyperparameter combinations
    and save each to disk.

    The grid covers:
      - 3 Skip-gram configs: baseline (100d, w5, n5), wider window, smaller dim
      - 3 CBOW configs:      matching the Skip-gram grid

    Args:
        sentences (list of list of str): Preprocessed tokenized sentences.
        vocab     (Vocabulary)         : Pre-built vocabulary object.

    Returns:
        tuple: (models dict, configs list)
            models  — label (str) → trained Word2Vec model
            configs — list of hyperparameter dictionaries used
    """
    # Encode sentences to integer index lists for training
    encoded = vocab.encode_sentences(sentences)
    print(f"\nEncoded {len(encoded)} sentences for training.\n")

    # Define the hyperparameter grid
    # Each entry has: architecture, embedding dim, window size, neg samples, label
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

        # Instantiate and train the model for this configuration
        model = Word2Vec(
            vocab,
            embedding_dim=cfg['dim'],
            window=cfg['window'],
            neg_samples=cfg['neg'],
            learning_rate=0.025,
            architecture=cfg['arch']
        )
        model.train(encoded, epochs=15, verbose=True)

        # Persist model weights to disk for later evaluation
        model.save(f"./model_{cfg['label']}.pkl")
        models[cfg['label']] = model

    return models, configs


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'os.getcwd()')

    # Import preprocessing utilities from Task 1
    from task1_preprocessing import build_corpus, compute_stats
    from iitj_corpus_raw import get_corpus

    # Load and preprocess the corpus
    documents = get_corpus()
    sentences = build_corpus(documents)
    stats = compute_stats(sentences, documents)

    print(f"Corpus: {stats['num_sentences']} sentences, "
          f"{stats['total_tokens']} tokens, vocab {stats['vocab_size']}")

    # Build vocabulary (min_count=2 filters hapax legomena)
    vocab = Vocabulary(min_count=2)
    vocab.build(sentences)

    # Train all experiment configurations
    models, configs = run_experiments(sentences, vocab)
    print("\nAll models trained successfully!")