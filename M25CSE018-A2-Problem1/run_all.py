"""

Runs ALL tasks end-to-end and generates every output file:
  - Task 1: Corpus preprocessing & statistics
  - Task 2: Word2Vec model training (18 hyperparameter configurations)
  - Task 3: Semantic analysis — nearest neighbours & analogy experiments
  - Task 4: Visualisation — PCA, t-SNE, cosine similarity heatmaps
  - Task 5: Gensim comparison — custom vs. production-grade framework

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — write figures to disk without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import collections
import random
import math
import sys
import os
import json
import warnings

# Suppress minor NumPy/Matplotlib deprecation warnings to keep output clean
warnings.filterwarnings('ignore')

# Fix random seeds so all results are reproducible across runs
random.seed(42)
np.random.seed(42)

# Make the local project directory importable (for iitj_corpus_raw, task modules)
sys.path.insert(0, os.getcwd())

# ─── LOAD MODULES ──────────────────────────────────────────────────────────────
from iitj_corpus_raw import get_corpus                              # Raw corpus data
from task1_preprocessing import build_corpus, compute_stats, save_cleaned_corpus  # Task 1
from task2_word2vec import Vocabulary, Word2Vec                     # Task 2 model

# Ensure the output directory exists before any file writes
os.makedirs('./outputs', exist_ok=True)

print("=" * 60)
print("IIT JODHPUR WORD2VEC ASSIGNMENT — FULL PIPELINE")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: PREPROCESSING
# Load raw documents, run the cleaning pipeline, and report corpus statistics.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TASK 1] Dataset Preparation & Preprocessing")
print("-" * 45)

# Load raw academic text documents from the IIT Jodhpur corpus
documents = get_corpus()

# Apply full preprocessing: lowercase, remove noise, tokenize, split into sentences
sentences = build_corpus(documents)

# Compute descriptive statistics for reporting
stats = compute_stats(sentences, documents)

print(f"  Documents  : {stats['num_documents']}")
print(f"  Sentences  : {stats['num_sentences']}")
print(f"  Tokens     : {stats['total_tokens']}")
print(f"  Vocab size : {stats['vocab_size']}")

# Write the cleaned, tokenized corpus to disk (one sentence per line)
save_cleaned_corpus(sentences, './outputs/cleaned_corpus.txt')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: WORD CLOUD (manual — no external wordcloud library)
# Visualises the 80 most frequent non-stop words as randomly placed text,
# with font size proportional to frequency.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PLOT] Generating Word Cloud (frequency-based)...")

# Additional stop words to filter from the word cloud (beyond the Task 1 set)
# These are high-frequency words that add little semantic content to the visualisation.
STOP_CLOUD = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'from','is','are','was','were','be','been','have','has','had','do','does',
    'did','will','would','could','should','that','this','these','those','it',
    'its','they','them','their','all','each','as','also','not','may','can',
    'such','both','more','than','if','any','been','there','which','who','how',
    'we','our','he','she','i','you','me','us','him','her','my','your','his',
    'so','very','too','just','than','then','when','where','what','per','into',
    'over','after','through','between','among','under','above','across'
}

freq = stats['freq']   # Counter object from Task 1

# Take the top 80 content words (excluding stop words and very short words)
cloud_words = [(w, c) for w, c in freq.most_common(200)
               if w not in STOP_CLOUD and len(w) > 2][:80]

# Create a dark background figure
fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0a0f1e')
ax.set_facecolor('#0a0f1e')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')   # Hide all axis elements — the canvas is the word cloud

max_count = cloud_words[0][1]   # Highest frequency (for normalisation)
min_count = cloud_words[-1][1]  # Lowest frequency in top-80

palette = ['#00d4ff', '#7b2fff', '#ff6b35', '#00ff88', '#ffcc02',
           '#ff3d71', '#a78bfa', '#34d399']   # Accent colour palette

placed = []     # Track placed words (for potential future collision detection)
random.seed(123)  # Separate seed for word placement (independent of model training seed)

for word, count in cloud_words:
    # Normalise frequency to [0, 1] for font size and opacity scaling
    norm  = (count - min_count) / max(max_count - min_count, 1)
    size  = int(8 + norm * 38)       # Font size: 8pt (rare) … 46pt (most frequent)
    color = random.choice(palette)   # Random colour from the palette
    alpha = 0.65 + norm * 0.35       # Opacity: slightly more opaque for frequent words

    # Place each word at a random position (simple approach without overlap detection)
    for _ in range(300):   # Up to 300 attempts (always breaks on first try here)
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        ax.text(x, y, word, fontsize=size, color=color, alpha=alpha,
                ha='center', va='center', fontweight='bold',
                fontfamily='DejaVu Sans')
        placed.append((x, y, word))
        break  # Place immediately without retrying (fast, but allows overlaps)

# Add title and subtitle annotations
ax.text(0.5, 0.97, 'IIT Jodhpur Corpus — Word Cloud',
        fontsize=16, color='white', ha='center', va='top',
        fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.02,
        f'Top 80 most frequent words  |  Vocab size: {stats["vocab_size"]}  |  Tokens: {stats["total_tokens"]}',
        fontsize=9, color='#888', ha='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('./outputs/wordcloud.png', dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
plt.close()
print("  Saved: wordcloud.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: WORD FREQUENCY BAR CHART
# Horizontal bar chart showing the top 25 most frequent content words.
# ─────────────────────────────────────────────────────────────────────────────

# Select top 25 content words (filter stop words, keep length > 2)
top_words  = [(w, c) for w, c in freq.most_common(300)
              if w not in STOP_CLOUD and len(w) > 2][:25]
words_bar  = [w for w, c in top_words]   # Just the word labels
counts_bar = [c for w, c in top_words]   # Just the counts

fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0a0f1e')
ax.set_facecolor('#111827')

# Plasma colormap gradient: each bar gets a colour from dark purple to yellow
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(words_bar)))
bars   = ax.barh(range(len(words_bar)), counts_bar,
                 color=colors, edgecolor='none', height=0.7)

ax.set_yticks(range(len(words_bar)))
ax.set_yticklabels(words_bar, fontsize=11, color='white')
ax.set_xlabel('Frequency', color='#aaa', fontsize=11)
ax.set_title('Top 25 Most Frequent Words — IIT Jodhpur Corpus',
             color='white', fontsize=14, fontweight='bold', pad=12)
ax.tick_params(colors='#aaa')
ax.spines[:].set_color('#333')

# Annotate each bar with the exact count value
for bar, count in zip(bars, counts_bar):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(count), va='center', color='#ccc', fontsize=9)

fig.patch.set_facecolor('#0a0f1e')
plt.tight_layout()
plt.savefig('./outputs/word_frequency.png', dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
plt.close()
print("  Saved: word_frequency.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: MODEL TRAINING
# Train 18 Word2Vec models covering 3 hyperparameter ablation groups
# (embedding dimension, context window, negative samples) × 2 architectures.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TASK 2] Training Word2Vec Models")
print("-" * 45)

# Build vocabulary with min_count=2 (removes hapax legomena)
vocab   = Vocabulary(min_count=2)
vocab.build(sentences)

# Convert sentences from token strings to integer index lists
encoded = vocab.encode_sentences(sentences)
print(f"  Encoded sentences: {len(encoded)}")
print(f"  Effective vocab  : {vocab.vocab_size}")

EPOCHS = 30   # All models trained for 30 epochs for a fair comparison

# ── Hyperparameter grid ──
# Each config varies ONE factor at a time (ablation design).
# Labels follow the pattern ARCH-factor{value} for easy grouping in plots.
configs = [
    # ── Skip-gram: vary embedding dimension (window=5, neg=5 fixed) ──
    {'arch': 'skipgram', 'dim':  50, 'window': 5, 'neg':  5, 'label': 'SG-dim50'},
    {'arch': 'skipgram', 'dim': 100, 'window': 5, 'neg':  5, 'label': 'SG-dim100'},  # baseline
    {'arch': 'skipgram', 'dim': 300, 'window': 5, 'neg':  5, 'label': 'SG-dim300'},

    # ── Skip-gram: vary context window (dim=100, neg=5 fixed) ──
    {'arch': 'skipgram', 'dim': 100, 'window': 2, 'neg':  5, 'label': 'SG-win2'},
    {'arch': 'skipgram', 'dim': 100, 'window': 5, 'neg':  5, 'label': 'SG-win5'},    # baseline
    {'arch': 'skipgram', 'dim': 100, 'window': 8, 'neg':  5, 'label': 'SG-win8'},

    # ── Skip-gram: vary negative samples (dim=100, window=5 fixed) ──
    {'arch': 'skipgram', 'dim': 100, 'window': 5, 'neg':  2, 'label': 'SG-neg2'},
    {'arch': 'skipgram', 'dim': 100, 'window': 5, 'neg':  5, 'label': 'SG-neg5'},    # baseline
    {'arch': 'skipgram', 'dim': 100, 'window': 5, 'neg': 10, 'label': 'SG-neg10'},

    # ── CBOW: vary embedding dimension ──
    {'arch': 'cbow',     'dim':  50, 'window': 5, 'neg':  5, 'label': 'CBOW-dim50'},
    {'arch': 'cbow',     'dim': 100, 'window': 5, 'neg':  5, 'label': 'CBOW-dim100'},# baseline
    {'arch': 'cbow',     'dim': 300, 'window': 5, 'neg':  5, 'label': 'CBOW-dim300'},

    # ── CBOW: vary context window ──
    {'arch': 'cbow',     'dim': 100, 'window': 2, 'neg':  5, 'label': 'CBOW-win2'},
    {'arch': 'cbow',     'dim': 100, 'window': 5, 'neg':  5, 'label': 'CBOW-win5'},  # baseline
    {'arch': 'cbow',     'dim': 100, 'window': 8, 'neg':  5, 'label': 'CBOW-win8'},

    # ── CBOW: vary negative samples ──
    {'arch': 'cbow',     'dim': 100, 'window': 5, 'neg':  2, 'label': 'CBOW-neg2'},
    {'arch': 'cbow',     'dim': 100, 'window': 5, 'neg':  5, 'label': 'CBOW-neg5'},  # baseline
    {'arch': 'cbow',     'dim': 100, 'window': 5, 'neg': 10, 'label': 'CBOW-neg10'},
]

models          = {}   # label → trained Word2Vec model
training_losses = {}   # label → list of per-epoch average losses

for cfg in configs:
    print(f"\n  Training: {cfg['label']}")

    # Instantiate a fresh model for this configuration
    model = Word2Vec(
        vocab,
        embedding_dim=cfg['dim'],
        window=cfg['window'],
        neg_samples=cfg['neg'],
        learning_rate=0.025,
        architecture=cfg['arch']
    )

    losses = []   # Store average loss after each epoch

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        count      = 0

        # Shuffle sentences each epoch to improve SGD convergence
        shuffled = encoded[:]
        random.shuffle(shuffled)

        # Linear LR decay: from 0.025 down to 0.025 * 0.0001 over all epochs
        lr = max(0.025 * (1 - epoch / EPOCHS), 0.025 * 0.0001)
        model.lr = lr

        for sent in shuffled:
            for i, center in enumerate(sent):
                # Dynamic window: sample radius uniformly from [1, max_window]
                # This naturally down-weights distant context words on average
                w = random.randint(1, model.window)

                # Collect context word indices within the dynamic window
                ctx_indices = [sent[j] for j in range(max(0, i - w),
                                                       min(len(sent), i + w + 1))
                               if j != i]
                if not ctx_indices:
                    continue

                # Sample negatives (excluding center and context words)
                neg_indices = model._get_negatives(
                    [center] + ctx_indices, model.neg_samples
                )

                if model.architecture == 'skipgram':
                    # Skip-gram: one weight update per (center, context_word) pair
                    for ctx_idx in ctx_indices:
                        epoch_loss += model._skipgram_update(center, ctx_idx, neg_indices)
                        count += 1
                else:
                    # CBOW: one update using the averaged context → center word
                    epoch_loss += model._cbow_update(ctx_indices, center, neg_indices)
                    count += 1

        avg = epoch_loss / max(count, 1)
        losses.append(avg)

        # Print progress every 5 epochs to show training is moving
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg:.4f} | LR: {lr:.6f}")

    models[cfg['label']]          = model
    training_losses[cfg['label']] = losses

print("\n  All models trained!")


# ─────────────────────────────────────────────────────────────────────────────
# BEST MODEL SELECTION
# Score each model on a combined metric:
#   combined_score = 1/(1 + final_loss) + 1/(1 + sim_mse)
# Higher score = lower loss AND lower similarity MSE.
# Best Skip-gram and best CBOW are selected separately.
# ─────────────────────────────────────────────────────────────────────────────

# Ground-truth word-pair similarity scores for the IIT Jodhpur academic domain
EVAL_PAIRS = [
    ('research',    'thesis',       0.90),
    ('student',     'faculty',      0.50),
    ('phd',         'doctoral',     0.95),
    ('exam',        'semester',     0.70),
    ('department',  'engineering',  0.70),
    ('btech',       'mtech',        0.60),
    ('professor',   'supervisor',   0.70),
    ('lab',         'research',     0.65),
    ('program',     'course',       0.60),
    ('grade',       'cgpa',         0.80),
]


def evaluate_similarity_mse(model, pairs):
    """
    Compute the Mean Squared Error between predicted cosine similarities
    and the expected (ground-truth) scores in `pairs`.

    Pairs involving OOV words are silently skipped.

    Args:
        model (Word2Vec): Trained model with a cosine_similarity() method.
        pairs (list)    : List of (word1, word2, expected_score) tuples.

    Returns:
        float: MSE score (lower is better), or inf if no pairs could be scored.
    """
    errors = []
    for w1, w2, expected in pairs:
        # Skip any pair where either word is not in this model's vocabulary
        if w1 not in model.vocab.word2idx or w2 not in model.vocab.word2idx:
            continue
        predicted = model.cosine_similarity(w1, w2)
        if predicted is not None:
            errors.append((predicted - expected) ** 2)
    return sum(errors) / len(errors) if errors else float('inf')


def combined_score(label, model, losses, pairs):
    """
    Compute the combined selection score for a model.
    Score = 1/(1 + final_epoch_loss) + 1/(1 + similarity_MSE)

    Both terms are in (0, 1] — higher is better.
    Summing them rewards models that are good on BOTH metrics.

    Args:
        label  (str)      : Model configuration label.
        model  (Word2Vec) : Trained model.
        losses (dict)     : label → list of per-epoch losses.
        pairs  (list)     : Evaluation word pairs.

    Returns:
        float: Combined score in (0, 2].
    """
    final_loss = losses[label][-1]                    # Loss of the last training epoch
    sim_mse    = evaluate_similarity_mse(model, pairs)
    return (1 / (1 + final_loss)) + (1 / (1 + sim_mse))


# Compute scores separately for Skip-gram and CBOW model families
sg_scores   = {l: combined_score(l, m, training_losses, EVAL_PAIRS)
               for l, m in models.items() if l.startswith('SG')}
cbow_scores = {l: combined_score(l, m, training_losses, EVAL_PAIRS)
               for l, m in models.items() if l.startswith('CBOW')}

# Select the configuration with the highest combined score in each family
best_sg_label   = max(sg_scores,   key=sg_scores.get)
best_cbow_label = max(cbow_scores, key=cbow_scores.get)

print(f"\n  ── Model Selection Results ──")
print(f"  {'Config':<22} {'Final Loss':>12} {'Sim MSE':>10} {'Score':>8}")
print(f"  {'-'*56}")

for l in sorted(sg_scores):
    fl     = training_losses[l][-1]
    mse    = evaluate_similarity_mse(models[l], EVAL_PAIRS)
    sc     = sg_scores[l]
    marker = '  ← BEST' if l == best_sg_label else ''
    print(f"  {l:<22} {fl:>12.4f} {mse:>10.4f} {sc:>8.4f}{marker}")

print(f"  {'-'*56}")

for l in sorted(cbow_scores):
    fl     = training_losses[l][-1]
    mse    = evaluate_similarity_mse(models[l], EVAL_PAIRS)
    sc     = cbow_scores[l]
    marker = '  ← BEST' if l == best_cbow_label else ''
    print(f"  {l:<22} {fl:>12.4f} {mse:>10.4f} {sc:>8.4f}{marker}")

print(f"\n  Best Skip-gram : {best_sg_label}  (score={sg_scores[best_sg_label]:.4f})")
print(f"  Best CBOW      : {best_cbow_label} (score={cbow_scores[best_cbow_label]:.4f})")

# Extract the winning models for downstream analysis
sg_model   = models[best_sg_label]
cbow_model = models[best_cbow_label]


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: TRAINING LOSS CURVES
# One subplot per architecture (Skip-gram / CBOW).
# Each config's loss curve is a dashed line; the best config is a thick solid line.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PLOT] Training loss curves...")

# 9 colours per architecture (one per config in that family)
SG_COLORS   = ['#00d4ff', '#7b2fff', '#ff6b35', '#00ff88', '#ffcc02',
                '#ff3d71', '#a78bfa', '#34d399', '#f97316']
CBOW_COLORS = ['#00ff88', '#ffcc02', '#ff3d71', '#00d4ff', '#7b2fff',
                '#ff6b35', '#a78bfa', '#34d399', '#f97316']

fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor='#0a0f1e')
fig.patch.set_facecolor('#0a0f1e')

for ax, arch, col_list in zip(axes, ['skipgram', 'cbow'], [SG_COLORS, CBOW_COLORS]):
    ax.set_facecolor('#111827')
    ax.spines[:].set_color('#333')
    ax.tick_params(colors='#aaa')
    ax.set_xlabel('Epoch', color='#aaa', fontsize=11)
    ax.set_ylabel('Average Loss', color='#aaa', fontsize=11)
    arch_label = 'Skip-gram' if arch == 'skipgram' else 'CBOW'
    ax.set_title(f'{arch_label} Training Loss', color='white', fontsize=13, fontweight='bold')

    # Filter to only the configs belonging to this architecture
    arch_losses = [(label, losses)
                   for label, losses in training_losses.items()
                   if (arch == 'skipgram' and label.startswith('SG')) or
                      (arch == 'cbow'     and label.startswith('CBOW'))]

    for (label, losses), c in zip(arch_losses, col_list):
        is_best = label in (best_sg_label, best_cbow_label)
        lw      = 3.0 if is_best else 1.5  # Thicker line for the best model
        ls      = '-'  if is_best else '--' # Solid vs. dashed

        display_label = f'{label} ★' if is_best else label
        ax.plot(range(1, len(losses) + 1), losses,
                color=c, linewidth=lw, linestyle=ls, label=display_label)

    ax.legend(facecolor='#1f2937', edgecolor='#444', labelcolor='white', fontsize=8)
    ax.grid(True, color='#222', linewidth=0.5)

plt.suptitle('Word2Vec Training Loss — IIT Jodhpur Corpus',
             color='white', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('./outputs/training_loss.png', dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
plt.close()
print("  Saved: training_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: SEMANTIC ANALYSIS
# Use the best Skip-gram and CBOW models to:
#   (a) retrieve top-5 nearest neighbours for key academic vocabulary
#   (b) run word analogy experiments using vec(b) - vec(a) + vec(c)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TASK 3] Semantic Analysis")
print("-" * 45)
print(f"  Using best Skip-gram : {best_sg_label}")
print(f"  Using best CBOW      : {best_cbow_label}")

# Five representative words from the IIT Jodhpur academic domain
query_words = ['research', 'student', 'phd', 'exam', 'department']

print("\n  ── Top-5 Nearest Neighbors (Skip-gram) ──")
sg_neighbors = {}
for word in query_words:
    if word in vocab.word2idx:
        neighbors = sg_model.most_similar(word, topn=5)
        sg_neighbors[word] = neighbors
        print(f"  {word}: " + ", ".join(f"{w}({s:.3f})" for w, s in neighbors))
    else:
        print(f"  '{word}' not in vocabulary")
        sg_neighbors[word] = []

print("\n  ── Top-5 Nearest Neighbors (CBOW) ──")
cbow_neighbors = {}
for word in query_words:
    if word in vocab.word2idx:
        neighbors = cbow_model.most_similar(word, topn=5)
        cbow_neighbors[word] = neighbors
        print(f"  {word}: " + ", ".join(f"{w}({s:.3f})" for w, s in neighbors))
    else:
        cbow_neighbors[word] = []

# ── Analogy Experiments ──
# Format: (word_a, word_b, word_c, description)
# Expected answer: vec(b) - vec(a) + vec(c) ≈ answer
print("\n  ── Analogy Experiments ──")
analogies = [
    ('ug',       'btech',  'pg',       "UG : BTech :: PG : ?"),
    ('student',  'exam',   'faculty',  "student : exam :: faculty : ?"),
    ('research', 'phd',    'teaching', "research : PhD :: teaching : ?"),
    ('mtech',    'thesis', 'btech',    "MTech : thesis :: BTech : ?"),
]

analogy_results = {}
for wa, wb, wc, desc in analogies:
    result_sg, msg_sg   = sg_model.analogy(wa, wb, wc)
    result_cb, msg_cb   = cbow_model.analogy(wa, wb, wc)
    analogy_results[desc] = {'sg': result_sg, 'cbow': result_cb, 'words': (wa, wb, wc)}

    print(f"\n  {desc}")
    sg_str = ", ".join(f"{w}({s:.3f})" for w, s in result_sg[:3]) if result_sg else msg_sg
    cb_str = ", ".join(f"{w}({s:.3f})" for w, s in result_cb[:3]) if result_cb else msg_cb
    print(f"    Skip-gram : {sg_str}")
    print(f"    CBOW      : {cb_str}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: VISUALISATION — DIMENSIONALITY REDUCTION & SIMILARITY HEATMAPS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TASK 4] Dimensionality Reduction & Visualization")
print("-" * 45)


def pca_2d(matrix):
    """
    Project an (N, D) embedding matrix to 2D using PCA (no sklearn).
    Computes the top-2 eigenvectors of the covariance matrix manually.

    Args:
        matrix (np.ndarray): Shape (N, D).

    Returns:
        np.ndarray: Shape (N, 2) projected coordinates.
    """
    mu       = matrix.mean(axis=0)
    centered = matrix - mu              # Zero-mean the data
    cov      = centered.T @ centered / len(matrix)  # Covariance matrix

    # Eigendecomposition: eigh is for symmetric matrices (numerically stable)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]   # Sort descending by explained variance

    # Project onto the top-2 principal components
    return centered @ eigenvectors[:, idx][:, :2]


def tsne_2d(matrix, perplexity=15, n_iter=500, lr=100):
    """
    Simplified t-SNE dimensionality reduction implemented in pure NumPy.
    Useful for visualising high-dimensional embeddings in 2D without sklearn.

    The algorithm:
      1. Compute pairwise squared Euclidean distances.
      2. Fit per-row Gaussian bandwidths (beta) to achieve the target perplexity.
      3. Symmetrize and normalize the high-dimensional probability matrix P.
      4. Initialize low-dimensional embedding Y randomly.
      5. Iteratively update Y using gradient descent on the KL divergence
         between P and the Student t-distribution Q in low-D space.

    Args:
        matrix     (np.ndarray): Shape (N, D) embedding matrix.
        perplexity (float)     : Effective number of neighbours. Default: 15.
        n_iter     (int)       : Number of gradient descent steps. Default: 500.
        lr         (float)     : Learning rate. Default: 100.

    Returns:
        np.ndarray: Shape (N, 2) t-SNE coordinates.
    """
    n   = len(matrix)

    # Standardize the data before running t-SNE for numerical stability
    mat = matrix - matrix.mean(axis=0)
    mat = mat / (mat.std() + 1e-10)

    # Pairwise squared Euclidean distances (N × N)
    D = np.sum((mat[:, None] - mat[None, :]) ** 2, axis=2)

    # ── Step 1: Compute the high-dimensional probability matrix P ──
    P = np.zeros((n, n))
    for i in range(n):
        beta  = 1.0    # Initial precision (1 / 2σ²)
        Di    = np.delete(D[i], i)   # Exclude self-distance
        target = np.log2(perplexity) # Binary entropy target

        # Binary search for the bandwidth beta that achieves target perplexity
        for _ in range(50):
            Pi     = np.exp(-Di * beta)
            Pi_sum = Pi.sum() or 1e-10
            H      = np.log(Pi_sum) + beta * np.dot(Di, Pi) / Pi_sum  # Shannon entropy
            if abs(H - target) < 1e-5:
                break
            beta *= 2 if H > target else 0.5   # Halve or double precision

        # Build the full probability row (including the zero self-element)
        Pi_full = np.zeros(n)
        Pi_full[np.arange(n) != i] = Pi / Pi_sum
        P[i] = Pi_full

    # Symmetrize: P_ij = (P_i|j + P_j|i) / (2N)  — ensures P is a joint distribution
    P     = (P + P.T) / (2 * n)
    P     = np.maximum(P, 1e-12)   # Clamp for numerical stability

    # ── Step 2: Initialize low-dimensional embedding Y ──
    Y     = np.random.randn(n, 2) * 0.01  # Small random init to prevent mode collapse
    dY    = np.zeros_like(Y)
    iY    = np.zeros_like(Y)               # Momentum term
    gains = np.ones_like(Y)               # Per-parameter adaptive gain

    # ── Step 3: Gradient descent on KL(P || Q) ──
    for t in range(n_iter):
        # Compute Student t-distribution Q in low-D space (heavy tails reduce crowding)
        D_Y   = np.sum((Y[:, None] - Y[None, :]) ** 2, axis=2)
        Q_num = 1.0 / (1.0 + D_Y)   # t-distribution numerator
        np.fill_diagonal(Q_num, 0)
        Q     = np.maximum(Q_num / Q_num.sum(), 1e-12)

        PQ = P - Q   # Element-wise difference for gradient

        # Compute gradient for each point i
        for i in range(n):
            dY[i] = 4 * np.sum(
                (PQ[i, :, None] * Q_num[i, :, None]) * (Y[i] - Y), axis=0
            )

        # Momentum schedule: 0.5 for early iterations, 0.8 after iteration 20
        momentum = 0.5 if t < 20 else 0.8

        # Adaptive gain: increase if gradient direction is consistent, decrease otherwise
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + \
                gains * 0.8   * ((dY > 0) == (iY > 0))
        gains = np.maximum(gains, 0.01)   # Floor gains to prevent stagnation

        # Update momentum and positions
        iY  = momentum * iY - lr * gains * dY
        Y  += iY
        Y  -= Y.mean(axis=0)   # Re-center after each step

    return Y


# ── Semantic word groups for PCA / t-SNE visualisation ──
# Each group has a name, a word list, and a colour for the scatter plot.
word_groups = {
    'Academic Programs': ['btech', 'mtech', 'phd', 'ug', 'pg', 'undergraduate',
                          'postgraduate', 'degree', 'program', 'course'],
    'Research':          ['research', 'thesis', 'dissertation', 'publication', 'paper',
                          'journal', 'conference', 'lab', 'experiment', 'findings'],
    'Faculty & Students':['student', 'faculty', 'professor', 'scholar', 'supervisor',
                          'mentor', 'advisor', 'researcher', 'fellow', 'postdoctoral'],
    'Examination':       ['exam', 'examination', 'grade', 'cgpa', 'semester', 'marks',
                          'assessment', 'quiz', 'attendance', 'result'],
    'Department':        ['department', 'engineering', 'science', 'mathematics', 'physics',
                          'chemistry', 'electrical', 'mechanical', 'civil', 'computer'],
}

group_colors = {
    'Academic Programs':  '#00d4ff',
    'Research':           '#ff6b35',
    'Faculty & Students': '#00ff88',
    'Examination':        '#ffcc02',
    'Department':         '#a78bfa',
}


def get_group_vectors(model, vocab, groups):
    """
    Collect embedding vectors for all in-vocabulary words from the group map.

    Args:
        model  (Word2Vec)  : Trained model with W_in weight matrix.
        vocab  (Vocabulary): Vocabulary with word2idx mapping.
        groups (dict)      : {group_name: [word_list]} mapping.

    Returns:
        tuple: (all_words list, all_group_labels list, all_colors list, vecs array)
    """
    all_words, all_labels, all_colors = [], [], []
    for group, words in groups.items():
        color = group_colors[group]
        for w in words:
            if w in vocab.word2idx:
                all_words.append(w)
                all_labels.append(group)
                all_colors.append(color)

    # Stack embedding vectors for all collected words
    vecs = np.array([model.W_in[vocab.word2idx[w]] for w in all_words])
    return all_words, all_labels, all_colors, vecs


# ── PCA Scatter Plots ──
print("  Running PCA projections...")
for model_name, model in [('Skip-gram', sg_model), ('CBOW', cbow_model)]:
    words, labels, colors, vecs = get_group_vectors(model, vocab, word_groups)

    if len(vecs) < 4:
        print(f"  Not enough words for {model_name}, skipping...")
        continue

    proj = pca_2d(vecs)   # Project embeddings to 2D

    fig, ax = plt.subplots(figsize=(12, 9), facecolor='#0a0f1e')
    ax.set_facecolor('#0d1117')
    ax.spines[:].set_color('#333')
    ax.tick_params(colors='#555')
    ax.set_title(f'PCA Projection of Word Embeddings — {model_name}\nIIT Jodhpur Corpus',
                 color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('PC 1', color='#888', fontsize=11)
    ax.set_ylabel('PC 2', color='#888', fontsize=11)

    for i, (word, color) in enumerate(zip(words, colors)):
        ax.scatter(proj[i, 0], proj[i, 1], color=color, s=120, alpha=0.9, zorder=3,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(word, (proj[i, 0], proj[i, 1]),
                    textcoords='offset points', xytext=(5, 4),
                    fontsize=8.5, color='white', alpha=0.85, fontweight='medium')

    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=legend_patches, loc='best',
              facecolor='#1f2937', edgecolor='#444',
              labelcolor='white', fontsize=9, title='Word Groups', title_fontsize=9)

    fig.patch.set_facecolor('#0a0f1e')
    plt.tight_layout()
    fname = f'./outputs/pca_{model_name.replace("-", "_").lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
    plt.close()
    print(f"  Saved: pca_{model_name.replace('-', '_').lower()}.png")


# ── t-SNE Scatter Plots ──
print("  Running t-SNE projections (this may take a minute)...")
for model_name, model in [('Skip-gram', sg_model), ('CBOW', cbow_model)]:
    words, labels, colors, vecs = get_group_vectors(model, vocab, word_groups)

    if len(vecs) < 4:
        continue

    try:
        # Use a conservative perplexity: min(10, N//2) to handle small word sets
        proj = tsne_2d(vecs, perplexity=min(10, len(vecs) // 2), n_iter=300)

        fig, ax = plt.subplots(figsize=(12, 9), facecolor='#0a0f1e')
        ax.set_facecolor('#0d1117')
        ax.spines[:].set_color('#333')
        ax.tick_params(colors='#555')
        ax.set_title(f't-SNE Projection of Word Embeddings — {model_name}\nIIT Jodhpur Corpus',
                     color='white', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('t-SNE Dim 1', color='#888', fontsize=11)
        ax.set_ylabel('t-SNE Dim 2', color='#888', fontsize=11)

        for i, (word, color) in enumerate(zip(words, colors)):
            ax.scatter(proj[i, 0], proj[i, 1], color=color, s=120, alpha=0.9, zorder=3,
                       edgecolors='white', linewidths=0.5)
            ax.annotate(word, (proj[i, 0], proj[i, 1]),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=8.5, color='white', alpha=0.85)

        legend_patches = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
        ax.legend(handles=legend_patches, loc='best',
                  facecolor='#1f2937', edgecolor='#444',
                  labelcolor='white', fontsize=9, title='Word Groups')

        fig.patch.set_facecolor('#0a0f1e')
        plt.tight_layout()
        fname = f'./outputs/tsne_{model_name.replace("-", "_").lower()}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
        plt.close()
        print(f"  Saved: tsne_{model_name.replace('-', '_').lower()}.png")

    except Exception as e:
        print(f"  t-SNE failed for {model_name}: {e}")


# ── Cosine Similarity Heatmaps ──
print("\n  Generating similarity heatmap...")

# Only include words that are actually present in the trained vocabulary
heatmap_words = [w for w in
                 ['research', 'phd', 'student', 'exam', 'btech', 'mtech',
                  'faculty', 'department', 'thesis', 'course', 'semester',
                  'professor', 'lab', 'grade', 'program']
                 if w in vocab.word2idx]

for model_name, model in [('Skip-gram', sg_model), ('CBOW', cbow_model)]:
    n          = len(heatmap_words)
    sim_matrix = np.zeros((n, n))

    # Build the full N×N pairwise cosine similarity matrix
    for i, w1 in enumerate(heatmap_words):
        for j, w2 in enumerate(heatmap_words):
            s = model.cosine_similarity(w1, w2)
            sim_matrix[i, j] = s if s is not None else 0.0

    fig, ax = plt.subplots(figsize=(10, 9), facecolor='#0a0f1e')
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0d1117')

    # Render as a false-colour image: dark = low similarity, bright = high similarity
    im = ax.imshow(sim_matrix, cmap='plasma', vmin=-0.2, vmax=1.0, aspect='auto')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(heatmap_words, rotation=45, ha='right', color='white', fontsize=9)
    ax.set_yticklabels(heatmap_words, color='white', fontsize=9)
    ax.set_title(f'Cosine Similarity Matrix — {model_name}\nIIT Jodhpur Embeddings',
                 color='white', fontsize=13, fontweight='bold', pad=12)

    # Annotate each cell with the numeric value
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if sim_matrix[i, j] < 0.6 else 'black')

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04,
                 label='Cosine Similarity').ax.yaxis.label.set_color('white')
    ax.spines[:].set_color('#333')
    plt.tight_layout()

    fname = f'./outputs/heatmap_{model_name.replace("-", "_").lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
    plt.close()
    print(f"  Saved: heatmap_{model_name.replace('-', '_').lower()}.png")


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER COMPARISON PLOT
# Three grouped bar charts: one per ablation axis.
# Each chart shows the average final training loss for SG vs. CBOW
# at each setting of the ablated hyperparameter.
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating hyperparameter comparison plot...")

# Extract the final (last-epoch) training loss for each config
final_losses = {label: losses[-1] for label, losses in training_losses.items()}

# Group configs by the ablated factor.
# Label format is always 'ARCH-groupkey' (e.g. 'SG-dim50', 'CBOW-win8', 'SG-neg10')
# Splitting on '-' gives ['ARCH', 'groupkey'] — avoids cross-group contamination.

dims = {'dim50': [], 'dim100': [], 'dim300': []}
for label, loss in final_losses.items():
    parts = label.split('-')
    if len(parts) == 2 and parts[1] in dims:
        dims[parts[1]].append((label, loss))

windows = {'win2': [], 'win5': [], 'win8': []}
for label, loss in final_losses.items():
    parts = label.split('-')
    if len(parts) == 2 and parts[1] in windows:
        windows[parts[1]].append((label, loss))

neg_samples = {'neg2': [], 'neg5': [], 'neg10': []}
for label, loss in final_losses.items():
    parts = label.split('-')
    if len(parts) == 2 and parts[1] in neg_samples:
        neg_samples[parts[1]].append((label, loss))


def bar_group(ax, groups, xlabel, title):
    """
    Draw a grouped bar chart comparing Skip-gram vs. CBOW average final loss
    for each value of a hyperparameter.

    Args:
        ax     (matplotlib.Axes): Target axes.
        groups (dict)           : {param_value: [(label, loss), ...]} dict.
        xlabel (str)            : X-axis label (name of the varied parameter).
        title  (str)            : Axes title.
    """
    ax.set_facecolor('#111827')
    ax.spines[:].set_color('#333')
    ax.tick_params(colors='#aaa')
    ax.set_xlabel(xlabel, color='#aaa')
    ax.set_ylabel('Final Loss', color='#aaa')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')

    labels_plot = list(groups.keys())
    sg_vals, cb_vals = [], []

    for key, items in groups.items():
        # Separate SG and CBOW results, then average (handles baselines appearing in both)
        sg_l = [l for lbl, l in items if lbl.startswith('SG')]
        cb_l = [l for lbl, l in items if lbl.startswith('CBOW')]
        sg_vals.append(np.mean(sg_l) if sg_l else 0)
        cb_vals.append(np.mean(cb_l) if cb_l else 0)

    x = np.arange(len(labels_plot))
    w = 0.35   # Bar width; two bars per group separated by their x-offsets

    ax.bar(x - w / 2, sg_vals, w, color='#00d4ff', label='Skip-gram', alpha=0.85)
    ax.bar(x + w / 2, cb_vals, w, color='#ff6b35', label='CBOW',      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, color='white')
    ax.legend(facecolor='#1f2937', edgecolor='#444', labelcolor='white', fontsize=9)
    ax.grid(True, axis='y', color='#222', linewidth=0.5)


fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0f1e')
fig.patch.set_facecolor('#0a0f1e')
fig.suptitle('Hyperparameter Comparison — Final Training Loss',
             color='white', fontsize=14, fontweight='bold')

bar_group(axes[0], dims,        'Embedding Dim',   'Effect of Embedding Dimension')
bar_group(axes[1], windows,     'Window Size',      'Effect of Context Window')
bar_group(axes[2], neg_samples, 'Negative Samples', 'Effect of Negative Samples')

plt.tight_layout()
plt.savefig('./outputs/hyperparam_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0f1e')
plt.close()
print("  Saved: hyperparam_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING VECTOR OUTPUT
# Save the 300-dimensional embedding for a chosen word to a text file.
# Always uses the SG-dim300 model (the highest-dimensional Skip-gram).
# ─────────────────────────────────────────────────────────────────────────────
CHOSEN_WORD = 'research'          # Word whose vector will be exported
model_300d  = models['SG-dim300'] # Use the 300-dimensional Skip-gram model

vec = model_300d.get_vector(CHOSEN_WORD)
if vec is not None:
    vec_str = ', '.join(f'{v:.4f}' for v in vec)
    print(f"\n300-dim embedding for '{CHOSEN_WORD}' (from SG-dim300):")
    print(f"{CHOSEN_WORD} - {vec_str}")

    with open('./outputs/embedding_300d.txt', 'w') as f:
        f.write(f"Word  : {CHOSEN_WORD}\n")
        f.write(f"Model : SG-dim300 (Skip-gram, dim=300, window=5, neg=5)\n")
        f.write(f"Dim   : 300\n")
        f.write(f"Corpus: IIT Jodhpur, vocab={vocab.vocab_size}\n\n")
        f.write(f"{CHOSEN_WORD} - {vec_str}\n")
    print("  Saved: embedding_300d.txt")
else:
    print(f"  '{CHOSEN_WORD}' not found in vocabulary.")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO JSON
# Persist all key numeric results for use in the written report.
# ─────────────────────────────────────────────────────────────────────────────
results = {
    'best_models': {
        'skipgram':     best_sg_label,
        'cbow':         best_cbow_label,
        'sg_score':     round(sg_scores[best_sg_label],                        4),
        'cbow_score':   round(cbow_scores[best_cbow_label],                    4),
        'sg_sim_mse':   round(evaluate_similarity_mse(sg_model,   EVAL_PAIRS), 4),
        'cbow_sim_mse': round(evaluate_similarity_mse(cbow_model, EVAL_PAIRS), 4),
    },
    'dataset_stats': {
        'num_documents':   stats['num_documents'],
        'num_sentences':   stats['num_sentences'],
        'total_tokens':    stats['total_tokens'],
        'vocab_size':      stats['vocab_size'],
        'effective_vocab': vocab.vocab_size,   # After min_count filter
    },
    'top_30_words': stats['top_50'][:30],
    'skipgram_neighbors': {w: [(n, round(s, 4)) for n, s in nbrs]
                           for w, nbrs in sg_neighbors.items()},
    'cbow_neighbors':     {w: [(n, round(s, 4)) for n, s in nbrs]
                           for w, nbrs in cbow_neighbors.items()},
    'analogies': {
        desc: {
            'words':         res['words'],
            'skipgram_top3': [(w, round(s, 4)) for w, s in res['sg'][:3]]   if res['sg']   else [],
            'cbow_top3':     [(w, round(s, 4)) for w, s in res['cbow'][:3]] if res['cbow'] else [],
        }
        for desc, res in analogy_results.items()
    },
    'final_losses': {k: round(v, 5) for k, v in final_losses.items()},
    'all_scores': {
        'skipgram': {l: round(s, 4) for l, s in sg_scores.items()},
        'cbow':     {l: round(s, 4) for l, s in cbow_scores.items()},
    },
}

with open('./outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n  Saved: results.json")
print("\n" + "=" * 60)
print("ALL TASKS COMPLETE — Outputs in ./outputs/")
print("=" * 60)


# =============================================================================
# TASK 5: GENSIM WORD2VEC TRAINING & COMPARISON WITH CUSTOM MODELS
# Trains Gensim models that mirror the best custom configurations,
# then runs head-to-head evaluation on neighbours, similarity MSE,
# analogy accuracy, training time, and four comparison visualisation plots.
# =============================================================================
print("\n" + "=" * 60)
print("[TASK 5] Gensim Word2Vec Training & Comparison")
print("=" * 60)

try:
    from gensim.models import Word2Vec as GensimW2V
    GENSIM_AVAILABLE = True
    print("  Gensim imported successfully.")
except ImportError:
    GENSIM_AVAILABLE = False
    print("  ERROR: gensim not installed. Run: pip install gensim")

if GENSIM_AVAILABLE:

    # Map label string → config dict so we can mirror the exact hyperparams in Gensim
    label_to_cfg = {cfg['label']: cfg for cfg in configs}
    sg_cfg       = label_to_cfg[best_sg_label]
    cbow_cfg     = label_to_cfg[best_cbow_label]

    # Gensim takes raw token lists (sentences) — no integer encoding needed
    print(f"\n  Mirroring best custom Skip-gram  : {best_sg_label}")
    print(f"  Mirroring best custom CBOW       : {best_cbow_label}")

    # ── Train Gensim Skip-gram ──
    # sg=1 → Skip-gram; hs=0 + negative=k → negative sampling (mirrors our custom impl)
    print(f"\n  Training Gensim Skip-gram  (dim={sg_cfg['dim']}, window={sg_cfg['window']}, neg={sg_cfg['neg']}) ...")
    gensim_sg = GensimW2V(
        sentences=sentences,
        vector_size=sg_cfg['dim'],
        window=sg_cfg['window'],
        min_count=2,
        sg=1,                      # Skip-gram architecture
        hs=0,                      # Use negative sampling (not hierarchical softmax)
        negative=sg_cfg['neg'],
        alpha=0.025,
        min_alpha=0.0001,
        epochs=EPOCHS,
        workers=1,                 # Single thread for reproducibility
        seed=42,
    )
    print("  Gensim Skip-gram training complete.")

    # ── Train Gensim CBOW ──
    # sg=0 → CBOW; all other parameters mirror the best custom CBOW config
    print(f"\n  Training Gensim CBOW  (dim={cbow_cfg['dim']}, window={cbow_cfg['window']}, neg={cbow_cfg['neg']}) ...")
    gensim_cbow = GensimW2V(
        sentences=sentences,
        vector_size=cbow_cfg['dim'],
        window=cbow_cfg['window'],
        min_count=2,
        sg=0,                      # CBOW architecture
        hs=0,
        negative=cbow_cfg['neg'],
        alpha=0.025,
        min_alpha=0.0001,
        epochs=EPOCHS,
        workers=1,
        seed=42,
    )
    print("  Gensim CBOW training complete.")


    # ─────────────────────────────────────────────────────────────────────────
    # COMPARISON HELPER FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def gensim_cosine(model, w1, w2):
        """
        Cosine similarity between two words using a Gensim Word2Vec model.

        Returns:
            float | None: Similarity score, or None if either word is OOV.
        """
        try:
            return float(model.wv.similarity(w1, w2))
        except KeyError:
            return None

    def gensim_most_similar(model, word, topn=5):
        """
        Top-N most similar words from a Gensim model.

        Returns:
            list of (word, score) tuples, or [] if word is OOV.
        """
        try:
            return model.wv.most_similar(word, topn=topn)
        except KeyError:
            return []

    def gensim_analogy(model, wa, wb, wc, topn=5):
        """
        Solve analogy wa:wb :: wc:? using Gensim vector arithmetic.
        Gensim API: positive=[wb, wc], negative=[wa] → vec(wb) - vec(wa) + vec(wc)

        Returns:
            tuple: (results list, message str)
        """
        try:
            results = model.wv.most_similar(positive=[wb, wc], negative=[wa], topn=topn)
            return results, "ok"
        except KeyError as e:
            return [], str(e)

    def gensim_sim_mse(model, pairs):
        """
        MSE of Gensim cosine similarities vs. ground-truth expected scores.
        OOV pairs are skipped.
        """
        errors = []
        for w1, w2, expected in pairs:
            s = gensim_cosine(model, w1, w2)
            if s is not None:
                errors.append((s - expected) ** 2)
        return sum(errors) / len(errors) if errors else float('inf')


    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION TABLE 1: NEAREST NEIGHBOURS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  COMPARISON: Nearest Neighbors")
    print("-" * 60)

    comparison_query_words = ['research', 'student', 'phd', 'exam', 'department']
    neighbor_comparison    = {}

    for word in comparison_query_words:
        row = {}

        # Retrieve top-5 neighbours from each of the four models
        row['custom_sg']   = sg_model.most_similar(word, topn=5) if word in sg_model.vocab.word2idx else []
        row['custom_cbow'] = cbow_model.most_similar(word, topn=5) if word in cbow_model.vocab.word2idx else []
        row['gensim_sg']   = gensim_most_similar(gensim_sg,   word, topn=5)
        row['gensim_cbow'] = gensim_most_similar(gensim_cbow, word, topn=5)

        neighbor_comparison[word] = row

        print(f"\n  Query: '{word}'")
        print(f"    Custom SG   : " + ", ".join(f"{w}({s:.3f})" for w, s in row['custom_sg']))
        print(f"    Gensim SG   : " + ", ".join(f"{w}({s:.3f})" for w, s in row['gensim_sg']))
        print(f"    Custom CBOW : " + ", ".join(f"{w}({s:.3f})" for w, s in row['custom_cbow']))
        print(f"    Gensim CBOW : " + ", ".join(f"{w}({s:.3f})" for w, s in row['gensim_cbow']))

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION TABLE 2: SIMILARITY MSE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  COMPARISON: Similarity MSE on Evaluation Pairs")
    print("-" * 60)

    mse_custom_sg   = evaluate_similarity_mse(sg_model,   EVAL_PAIRS)
    mse_custom_cbow = evaluate_similarity_mse(cbow_model, EVAL_PAIRS)
    mse_gensim_sg   = gensim_sim_mse(gensim_sg,   EVAL_PAIRS)
    mse_gensim_cbow = gensim_sim_mse(gensim_cbow, EVAL_PAIRS)

    print(f"\n  {'Model':<25} {'Sim MSE':>10}  {'Notes'}")
    print(f"  {'-'*55}")
    print(f"  {'Custom Skip-gram':<25} {mse_custom_sg:>10.4f}  ({best_sg_label})")
    print(f"  {'Gensim Skip-gram':<25} {mse_gensim_sg:>10.4f}")
    print(f"  {'Custom CBOW':<25} {mse_custom_cbow:>10.4f}  ({best_cbow_label})")
    print(f"  {'Gensim CBOW':<25} {mse_gensim_cbow:>10.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION TABLE 3: ANALOGY ACCURACY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  COMPARISON: Analogy Tasks")
    print("-" * 60)

    analogy_comparison = {}
    for wa, wb, wc, desc in analogies:
        res_gsg,  _ = gensim_analogy(gensim_sg,   wa, wb, wc)
        res_gcbow, _ = gensim_analogy(gensim_cbow, wa, wb, wc)
        analogy_comparison[desc] = {
            'gensim_sg':   res_gsg,
            'gensim_cbow': res_gcbow,
        }

        print(f"\n  {desc}")
        # Format top-3 predictions for each of the four models
        sg_str   = ", ".join(f"{w}({s:.3f})" for w, s in analogy_results[desc]['sg'][:3])   or "N/A"
        cb_str   = ", ".join(f"{w}({s:.3f})" for w, s in analogy_results[desc]['cbow'][:3]) or "N/A"
        gsg_str  = ", ".join(f"{w}({s:.3f})" for w, s in res_gsg[:3])    or "N/A"
        gcb_str  = ", ".join(f"{w}({s:.3f})" for w, s in res_gcbow[:3])  or "N/A"
        print(f"    Custom SG   : {sg_str}")
        print(f"    Gensim SG   : {gsg_str}")
        print(f"    Custom CBOW : {cb_str}")
        print(f"    Gensim CBOW : {gcb_str}")


    # ─────────────────────────────────────────────────────────────────────────
    # PLOT A — SIMILARITY MSE BAR CHART
    # Side-by-side bars comparing MSE for all four models.
    # Lower MSE = predictions closer to human-annotated similarity scores.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  [PLOT] Generating MSE comparison bar chart...")

    model_names_bar = ['Custom\nSkip-gram', 'Gensim\nSkip-gram',
                       'Custom\nCBOW', 'Gensim\nCBOW']
    mse_values      = [mse_custom_sg, mse_gensim_sg, mse_custom_cbow, mse_gensim_cbow]
    bar_colors      = ['#00d4ff', '#7b2fff', '#ff6b35', '#ffcc02']

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0f1e')
    ax.set_facecolor('#111827')
    ax.spines[:].set_color('#333')
    ax.tick_params(colors='#aaa')
    bars = ax.bar(model_names_bar, mse_values, color=bar_colors, edgecolor='none',
                  width=0.55, alpha=0.88)

    # Annotate each bar with the numeric MSE value
    for bar, val in zip(bars, mse_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=11,
                fontweight='bold')

    ax.set_ylabel('Similarity MSE  (lower = better)', color='#aaa', fontsize=11)
    ax.set_title('Custom vs. Gensim Word2Vec — Similarity MSE Comparison\nIIT Jodhpur Corpus',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, axis='y', color='#222', linewidth=0.5)
    fig.patch.set_facecolor('#0a0f1e')
    plt.tight_layout()
    plt.savefig('./outputs/comparison_mse_bar.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0f1e')
    plt.close()
    print("  Saved: comparison_mse_bar.png")


    # ─────────────────────────────────────────────────────────────────────────
    # PLOT B — 4-PANEL COSINE SIMILARITY HEATMAP
    # One panel per model: Custom SG, Gensim SG, Custom CBOW, Gensim CBOW.
    # Same word set on both axes so panels are directly comparable.
    # ─────────────────────────────────────────────────────────────────────────
    print("  [PLOT] Generating 4-panel similarity heatmap comparison...")

    # Use the same 10 core academic words for all four panels
    compare_words = [w for w in
                     ['research', 'phd', 'student', 'exam', 'btech', 'mtech',
                      'faculty', 'department', 'thesis', 'course']
                     if w in vocab.word2idx]
    n = len(compare_words)

    def build_sim_matrix_custom(model, words):
        """Build N×N cosine similarity matrix using the custom model."""
        mat = np.zeros((len(words), len(words)))
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                s = model.cosine_similarity(w1, w2)
                mat[i, j] = s if s is not None else 0.0
        return mat

    def build_sim_matrix_gensim(model, words):
        """Build N×N cosine similarity matrix using a Gensim model."""
        mat = np.zeros((len(words), len(words)))
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                s = gensim_cosine(model, w1, w2)
                mat[i, j] = s if s is not None else 0.0
        return mat

    # Compute all four similarity matrices
    mats = {
        f'Custom SG\n({best_sg_label})':    build_sim_matrix_custom(sg_model,    compare_words),
        f'Gensim SG':                        build_sim_matrix_gensim(gensim_sg,   compare_words),
        f'Custom CBOW\n({best_cbow_label})': build_sim_matrix_custom(cbow_model,  compare_words),
        f'Gensim CBOW':                      build_sim_matrix_gensim(gensim_cbow, compare_words),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='#0a0f1e')
    fig.patch.set_facecolor('#0a0f1e')
    fig.suptitle('Cosine Similarity Heatmaps — Custom vs. Gensim Word2Vec\nIIT Jodhpur Corpus',
                 color='white', fontsize=15, fontweight='bold', y=1.01)

    for ax, (title, mat) in zip(axes.flat, mats.items()):
        ax.set_facecolor('#0d1117')
        im = ax.imshow(mat, cmap='plasma', vmin=-0.2, vmax=1.0, aspect='auto')

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(compare_words, rotation=45, ha='right', color='white', fontsize=8)
        ax.set_yticklabels(compare_words, color='white', fontsize=8)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color='white' if mat[i, j] < 0.6 else 'black')

        ax.spines[:].set_color('#333')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04).ax.yaxis.label.set_color('white')

    plt.tight_layout()
    plt.savefig('./outputs/comparison_heatmaps.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0f1e')
    plt.close()
    print("  Saved: comparison_heatmaps.png")


    # ─────────────────────────────────────────────────────────────────────────
    # PLOT C — 4-PANEL PCA COMPARISON
    # Side-by-side PCA scatter plots for Custom SG, Gensim SG, Custom CBOW, Gensim CBOW.
    # Semantic grouping by colour lets us compare cluster structure across frameworks.
    # ─────────────────────────────────────────────────────────────────────────
    print("  [PLOT] Generating 4-panel PCA comparison...")

    def get_vectors_gensim(gensim_model, groups):
        """
        Collect embedding vectors from a Gensim model for the word group map.

        Args:
            gensim_model: Trained Gensim Word2Vec model.
            groups (dict): {group_name: [word_list]} mapping.

        Returns:
            tuple: (all_words, all_labels, all_colors, vecs array)
        """
        all_words, all_labels, all_colors = [], [], []
        for group, words in groups.items():
            color = group_colors[group]
            for w in words:
                try:
                    _ = gensim_model.wv[w]   # Check if word is in vocabulary
                    all_words.append(w)
                    all_labels.append(group)
                    all_colors.append(color)
                except KeyError:
                    pass   # Skip OOV words silently

        vecs = np.array([gensim_model.wv[w] for w in all_words])
        return all_words, all_labels, all_colors, vecs

    # Define the four panels: (title, source_type, model_object)
    panel_configs = [
        (f'Custom Skip-gram\n({best_sg_label})',   'custom',  sg_model),
        ('Gensim Skip-gram',                        'gensim',  gensim_sg),
        (f'Custom CBOW\n({best_cbow_label})',       'custom',  cbow_model),
        ('Gensim CBOW',                             'gensim',  gensim_cbow),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='#0a0f1e')
    fig.patch.set_facecolor('#0a0f1e')
    fig.suptitle('PCA Projections — Custom vs. Gensim Word2Vec\nIIT Jodhpur Corpus',
                 color='white', fontsize=15, fontweight='bold', y=1.01)

    for ax, (title, kind, model_obj) in zip(axes.flat, panel_configs):
        ax.set_facecolor('#0d1117')
        ax.spines[:].set_color('#333')
        ax.tick_params(colors='#555')
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('PC 1', color='#888', fontsize=10)
        ax.set_ylabel('PC 2', color='#888', fontsize=10)

        # Fetch vectors using the appropriate helper for this model type
        if kind == 'custom':
            words_p, labels_p, colors_p, vecs_p = get_group_vectors(model_obj, vocab, word_groups)
        else:
            words_p, labels_p, colors_p, vecs_p = get_vectors_gensim(model_obj, word_groups)

        if len(vecs_p) < 4:
            ax.text(0.5, 0.5, 'Insufficient words', color='white',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Project to 2D and render the scatter plot
        proj_p = pca_2d(vecs_p)
        for i, (word, color) in enumerate(zip(words_p, colors_p)):
            ax.scatter(proj_p[i, 0], proj_p[i, 1], color=color, s=100, alpha=0.9, zorder=3,
                       edgecolors='white', linewidths=0.4)
            ax.annotate(word, (proj_p[i, 0], proj_p[i, 1]),
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=7.5, color='white', alpha=0.85)

        legend_patches = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
        ax.legend(handles=legend_patches, loc='best',
                  facecolor='#1f2937', edgecolor='#444',
                  labelcolor='white', fontsize=7, title='Word Groups', title_fontsize=7)
        ax.grid(True, color='#1a1a2e', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('./outputs/comparison_pca.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0f1e')
    plt.close()
    print("  Saved: comparison_pca.png")


    # ─────────────────────────────────────────────────────────────────────────
    # PLOT D — NEAREST NEIGHBOUR OVERLAP BAR CHART
    # For each query word, count how many of the custom model's top-5 neighbours
    # also appear in the Gensim model's top-5 (0 = no agreement, 5 = perfect match).
    # ─────────────────────────────────────────────────────────────────────────
    print("  [PLOT] Generating neighbor overlap chart...")

    def overlap(list_a, list_b):
        """
        Count how many words appear in both top-N lists.

        Args:
            list_a, list_b (list of (word, score)): Neighbour lists from two models.

        Returns:
            int: Number of words in common.
        """
        set_a = {w for w, _ in list_a}
        set_b = {w for w, _ in list_b}
        return len(set_a & set_b)

    query_words_overlap = comparison_query_words

    # Compute overlap counts for SG pair and CBOW pair
    overlap_sg   = [overlap(neighbor_comparison[w]['custom_sg'],
                            neighbor_comparison[w]['gensim_sg'])
                    for w in query_words_overlap]
    overlap_cbow = [overlap(neighbor_comparison[w]['custom_cbow'],
                            neighbor_comparison[w]['gensim_cbow'])
                    for w in query_words_overlap]

    x_pos  = np.arange(len(query_words_overlap))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(11, 6), facecolor='#0a0f1e')
    ax.set_facecolor('#111827')
    ax.spines[:].set_color('#333')
    ax.tick_params(colors='#aaa')

    ax.bar(x_pos - width / 2, overlap_sg,   width, color='#00d4ff', alpha=0.88, label='Skip-gram')
    ax.bar(x_pos + width / 2, overlap_cbow, width, color='#ff6b35', alpha=0.88, label='CBOW')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(query_words_overlap, color='white', fontsize=11)
    ax.set_ylabel('Top-5 Neighbor Overlap (out of 5)', color='#aaa', fontsize=11)
    ax.set_title('Neighbor Agreement: Custom vs. Gensim Word2Vec\n(Higher = More Agreement)',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 5.5)

    # Reference line at 5 (perfect agreement)
    ax.axhline(y=5, color='#555', linestyle='--', linewidth=1, label='Max (5/5)')
    ax.legend(facecolor='#1f2937', edgecolor='#444', labelcolor='white', fontsize=10)
    ax.grid(True, axis='y', color='#222', linewidth=0.5)

    # Annotate each bar with its integer overlap count
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08,
                    str(int(h)), ha='center', va='bottom', color='white', fontsize=10)

    fig.patch.set_facecolor('#0a0f1e')
    plt.tight_layout()
    plt.savefig('./outputs/comparison_neighbor_overlap.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0f1e')
    plt.close()
    print("  Saved: comparison_neighbor_overlap.png")


    # ─────────────────────────────────────────────────────────────────────────
    # UPDATE results.json WITH GENSIM COMPARISON DATA
    # ─────────────────────────────────────────────────────────────────────────
    with open('./outputs/results.json', 'r') as f:
        results = json.load(f)

    # Append a new top-level key with all Task 5 comparison metrics
    results['gensim_comparison'] = {
        'gensim_sg_config': {
            'dim': sg_cfg['dim'], 'window': sg_cfg['window'],
            'neg': sg_cfg['neg'], 'epochs': EPOCHS,
        },
        'gensim_cbow_config': {
            'dim': cbow_cfg['dim'], 'window': cbow_cfg['window'],
            'neg': cbow_cfg['neg'], 'epochs': EPOCHS,
        },
        'similarity_mse': {
            'custom_sg':   round(mse_custom_sg,   4),
            'gensim_sg':   round(mse_gensim_sg,   4),
            'custom_cbow': round(mse_custom_cbow,  4),
            'gensim_cbow': round(mse_gensim_cbow,  4),
        },
        'neighbor_overlap': {
            w: {'sg': overlap_sg[i], 'cbow': overlap_cbow[i]}
            for i, w in enumerate(query_words_overlap)
        },
        'analogy_comparison': {
            desc: {
                'gensim_sg_top3':   [(w, round(s, 4)) for w, s in analogy_comparison[desc]['gensim_sg'][:3]],
                'gensim_cbow_top3': [(w, round(s, 4)) for w, s in analogy_comparison[desc]['gensim_cbow'][:3]],
            }
            for desc in analogy_comparison
        }
    }

    with open('./outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Updated: results.json (added gensim_comparison key)")


    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    avg_ol_sg   = sum(overlap_sg)   / len(overlap_sg)
    avg_ol_cbow = sum(overlap_cbow) / len(overlap_cbow)

    print("\n" + "=" * 60)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<26} {'Sim MSE':>10}  {'Avg Neighbor Overlap':>22}")
    print(f"  {'-'*62}")
    print(f"  {'Custom Skip-gram':<26} {mse_custom_sg:>10.4f}  {'— (baseline)':>22}")
    print(f"  {'Gensim Skip-gram':<26} {mse_gensim_sg:>10.4f}  {avg_ol_sg:>19.1f}/5")
    print(f"  {'Custom CBOW':<26} {mse_custom_cbow:>10.4f}  {'— (baseline)':>22}")
    print(f"  {'Gensim CBOW':<26} {mse_gensim_cbow:>10.4f}  {avg_ol_cbow:>19.1f}/5")
    print("=" * 60)

print("\n" + "=" * 60)
print("TASK 5 COMPLETE — New outputs in ./outputs/:")
print("  comparison_mse_bar.png")
print("  comparison_heatmaps.png")
print("  comparison_pca.png")
print("  comparison_neighbor_overlap.png")
print("  results.json  (updated with gensim_comparison key)")
print("=" * 60)