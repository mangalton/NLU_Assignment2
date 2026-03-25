"""
TASK 5: FRAMEWORK COMPARISON
Compares custom NumPy Word2Vec implementation against Gensim Word2Vec
on the same IIT Jodhpur corpus using identical hyperparameters.

Run this script from your project folder:
    pip install gensim
    python3 task5_comparison.py

Outputs (saved to ./outputs/):
    comparison_neighbors.png     — top-5 neighbour comparison bar chart
    comparison_heatmap.png       — side-by-side cosine similarity heatmaps
    comparison_pca.png           — side-by-side PCA projections (all 4 models)
    comparison_tsne.png          — side-by-side t-SNE projections (all 4 models)
    comparison_analogy.png       — analogy score comparison chart
    comparison_results.json      — all numeric results for report
"""

import os, sys, json, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.getcwd())

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec as GensimW2V
except ImportError:
    print("ERROR: gensim not installed. Run: pip install gensim")
    sys.exit(1)

from iitj_corpus_raw import get_corpus
from task1_preprocessing import build_corpus
from task2_word2vec import Vocabulary, Word2Vec as CustomW2V

os.makedirs('./outputs', exist_ok=True)

# ── Load corpus ────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 5 — Framework Comparison: Custom vs Gensim Word2Vec")
print("=" * 60)

print("\n[1] Loading and preprocessing corpus...")
docs      = get_corpus()
sentences = build_corpus(docs)
print(f"    Sentences: {len(sentences)}, "
      f"Tokens: {sum(len(s) for s in sentences):,}")

# ── Best hyperparameters (from results.json) ───────────────────────────────────
# Best models were SG-neg2 and CBOW-neg2:
#   dim=100, window=5, neg_samples=2
BEST_DIM    = 100
BEST_WINDOW = 5
BEST_NEG    = 2
EPOCHS      = 30

print(f"\n    Using best hyperparams: dim={BEST_DIM}, window={BEST_WINDOW}, "
      f"neg={BEST_NEG}, epochs={EPOCHS}")

# ── Build vocab for custom model ───────────────────────────────────────────────
vocab   = Vocabulary(min_count=2)
vocab.build(sentences)
encoded = vocab.encode_sentences(sentences)
print(f"    Vocabulary: {vocab.vocab_size} words")

# ─────────────────────────────────────────────────────────────
# TRAIN CUSTOM MODELS
# ─────────────────────────────────────────────────────────────
def train_custom(arch, dim, window, neg, epochs):
    model  = CustomW2V(vocab, embedding_dim=dim, window=window,
                       neg_samples=neg, learning_rate=0.025, architecture=arch)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        count      = 0
        shuffled   = encoded[:]
        random.shuffle(shuffled)
        lr = max(0.025 * (1 - epoch / epochs), 0.025 * 0.0001)
        model.lr = lr
        for sent in shuffled:
            for i, center in enumerate(sent):
                w = random.randint(1, model.window)
                ctx = [sent[j] for j in range(max(0, i-w),
                                               min(len(sent), i+w+1)) if j != i]
                if not ctx:
                    continue
                negs = model._get_negatives([center]+ctx, model.neg_samples)
                if arch == 'skipgram':
                    for c in ctx:
                        epoch_loss += model._skipgram_update(center, c, negs)
                        count += 1
                else:
                    epoch_loss += model._cbow_update(ctx, center, negs)
                    count += 1
        losses.append(epoch_loss / max(count, 1))
        if (epoch+1) % 10 == 0:
            print(f"      Epoch {epoch+1:2d}/{epochs} | loss={losses[-1]:.4f}")
    return model, losses

print("\n[2] Training Custom Skip-gram...")
t0 = time.time()
custom_sg, custom_sg_losses = train_custom('skipgram', BEST_DIM, BEST_WINDOW, BEST_NEG, EPOCHS)
custom_sg_time = time.time() - t0
print(f"    Done in {custom_sg_time:.1f}s | final loss={custom_sg_losses[-1]:.4f}")

print("\n[3] Training Custom CBOW...")
t0 = time.time()
custom_cb, custom_cb_losses = train_custom('cbow', BEST_DIM, BEST_WINDOW, BEST_NEG, EPOCHS)
custom_cb_time = time.time() - t0
print(f"    Done in {custom_cb_time:.1f}s | final loss={custom_cb_losses[-1]:.4f}")

# ─────────────────────────────────────────────────────────────
# TRAIN GENSIM MODELS
# ─────────────────────────────────────────────────────────────
print("\n[4] Training Gensim Skip-gram...")
t0 = time.time()
gensim_sg = GensimW2V(
    sentences=sentences,
    vector_size=BEST_DIM,
    window=BEST_WINDOW,
    negative=BEST_NEG,
    sg=1,              # 1 = skip-gram
    hs=0,              # 0 = negative sampling
    min_count=2,
    epochs=EPOCHS,
    seed=42,
    workers=1,         # single thread for reproducibility
    alpha=0.025,
    min_alpha=0.0001,
)
gensim_sg_time = time.time() - t0
print(f"    Done in {gensim_sg_time:.1f}s")

print("\n[5] Training Gensim CBOW...")
t0 = time.time()
gensim_cb = GensimW2V(
    sentences=sentences,
    vector_size=BEST_DIM,
    window=BEST_WINDOW,
    negative=BEST_NEG,
    sg=0,              # 0 = CBOW
    hs=0,
    min_count=2,
    epochs=EPOCHS,
    seed=42,
    workers=1,
    alpha=0.025,
    min_alpha=0.0001,
)
gensim_cb_time = time.time() - t0
print(f"    Done in {gensim_cb_time:.1f}s")

# ─────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────
QUERY_WORDS = ['research', 'student', 'phd', 'exam', 'department']

ANALOGY_TESTS = [
    # (a, b, c, expected_answer, description)
    ('btech',    'undergraduate', 'mtech',    'postgraduate',  'BTech:UG :: MTech:?'),
    ('phd',      'doctoral',      'mtech',    'postgraduate',  'PhD:doctoral :: MTech:?'),
    ('research', 'phd',           'teaching', 'btech',         'research:PhD :: teaching:?'),
    ('mtech',    'thesis',        'btech',    'project',       'MTech:thesis :: BTech:?'),
    ('student',  'exam',          'faculty',  'evaluation',    'student:exam :: faculty:?'),
]

EVAL_PAIRS = [
    ('research', 'thesis',       0.90),
    ('student',  'faculty',      0.50),
    ('phd',      'doctoral',     0.95),
    ('exam',     'semester',     0.70),
    ('department','engineering', 0.70),
    ('btech',    'mtech',        0.60),
    ('professor','supervisor',   0.70),
    ('lab',      'research',     0.65),
    ('program',  'course',       0.60),
    ('grade',    'cgpa',         0.80),
]

def custom_neighbors(model, word, topn=5):
    if word not in model.vocab.word2idx:
        return []
    return model.most_similar(word, topn=topn)

def gensim_neighbors(model, word, topn=5):
    if word not in model.wv:
        return []
    return [(w, float(s)) for w, s in model.wv.most_similar(word, topn=topn)]

def custom_cosine(model, w1, w2):
    return model.cosine_similarity(w1, w2)

def gensim_cosine(model, w1, w2):
    if w1 not in model.wv or w2 not in model.wv:
        return None
    return float(model.wv.similarity(w1, w2))

def custom_analogy(model, a, b, c):
    results, msg = model.analogy(a, b, c, topn=10)
    return [w for w, _ in results]

def gensim_analogy(model, a, b, c):
    try:
        if a not in model.wv or b not in model.wv or c not in model.wv:
            return []
        results = model.wv.most_similar(positive=[b, c], negative=[a], topn=10)
        return [w for w, _ in results]
    except:
        return []

def sim_mse(get_sim_fn, pairs):
    errors = []
    for w1, w2, expected in pairs:
        s = get_sim_fn(w1, w2)
        if s is not None:
            errors.append((s - expected) ** 2)
    return float(np.mean(errors)) if errors else float('inf')

def analogy_hits(get_analogy_fn, tests):
    hits, total = 0, 0
    details = []
    for a, b, c, expected, desc in tests:
        results = get_analogy_fn(a, b, c)
        hit     = expected in results[:5]
        in_top1 = results[0] == expected if results else False
        details.append({
            'analogy':  desc,
            'expected': expected,
            'got':      results[:3],
            'hit_top5': hit,
            'hit_top1': in_top1,
        })
        if results:            # only count if words were in vocab
            hits  += int(hit)
            total += 1
    acc = hits / total if total > 0 else 0.0
    return acc, details

# ─────────────────────────────────────────────────────────────
# RUN ALL EVALUATIONS
# ─────────────────────────────────────────────────────────────
print("\n[6] Running evaluations...")

# Neighbours
results_neighbors = {}
for word in QUERY_WORDS:
    results_neighbors[word] = {
        'custom_sg':  custom_neighbors(custom_sg,  word),
        'custom_cb':  custom_neighbors(custom_cb,  word),
        'gensim_sg':  gensim_neighbors(gensim_sg,  word),
        'gensim_cb':  gensim_neighbors(gensim_cb,  word),
    }

# Similarity MSE
mse_custom_sg = sim_mse(lambda w1,w2: custom_cosine(custom_sg, w1, w2), EVAL_PAIRS)
mse_custom_cb = sim_mse(lambda w1,w2: custom_cosine(custom_cb, w1, w2), EVAL_PAIRS)
mse_gensim_sg = sim_mse(lambda w1,w2: gensim_cosine(gensim_sg, w1, w2), EVAL_PAIRS)
mse_gensim_cb = sim_mse(lambda w1,w2: gensim_cosine(gensim_cb, w1, w2), EVAL_PAIRS)

# Analogy accuracy
acc_custom_sg, det_custom_sg = analogy_hits(lambda a,b,c: custom_analogy(custom_sg, a,b,c), ANALOGY_TESTS)
acc_custom_cb, det_custom_cb = analogy_hits(lambda a,b,c: custom_analogy(custom_cb, a,b,c), ANALOGY_TESTS)
acc_gensim_sg, det_gensim_sg = analogy_hits(lambda a,b,c: gensim_analogy(gensim_sg, a,b,c), ANALOGY_TESTS)
acc_gensim_cb, det_gensim_cb = analogy_hits(lambda a,b,c: gensim_analogy(gensim_cb, a,b,c), ANALOGY_TESTS)

print(f"\n    Similarity MSE  — Custom SG: {mse_custom_sg:.4f} | Gensim SG: {mse_gensim_sg:.4f}")
print(f"    Similarity MSE  — Custom CB: {mse_custom_cb:.4f} | Gensim CB: {mse_gensim_cb:.4f}")
print(f"    Analogy Top-5   — Custom SG: {acc_custom_sg:.2f}  | Gensim SG: {acc_gensim_sg:.2f}")
print(f"    Analogy Top-5   — Custom CB: {acc_custom_cb:.2f}  | Gensim CB: {acc_gensim_cb:.2f}")

# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────
DARK   = '#0a0f1e'
MID    = '#111827'
COLORS = {
    'custom_sg': '#00d4ff',
    'custom_cb': '#ff6b35',
    'gensim_sg': '#00ff88',
    'gensim_cb': '#ffcc02',
}
LABELS = {
    'custom_sg': 'Custom Skip-gram',
    'custom_cb': 'Custom CBOW',
    'gensim_sg': 'Gensim Skip-gram',
    'gensim_cb': 'Gensim CBOW',
}

def ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(MID)
    ax.spines[:].set_color('#333')
    ax.tick_params(colors='#aaa')
    if title:  ax.set_title(title,  color='white', fontsize=10, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color='#aaa',  fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color='#aaa',  fontsize=9)
    ax.grid(True, color='#222', linewidth=0.4)

def pca_2d(mat):
    mu  = mat.mean(axis=0)
    cen = mat - mu
    cov = cen.T @ cen / len(mat)
    ev, evec = np.linalg.eigh(cov)
    idx = np.argsort(ev)[::-1]
    return cen @ evec[:, idx][:, :2]

# ─────────────────────────────────────────────────────────────
# PLOT 1 — NEAREST NEIGHBOUR COMPARISON
# ─────────────────────────────────────────────────────────────
print("\n[7] Generating comparison plots...")

fig, axes = plt.subplots(len(QUERY_WORDS), 4, figsize=(20, 3.5*len(QUERY_WORDS)),
                          facecolor=DARK)
fig.patch.set_facecolor(DARK)
fig.suptitle('Nearest Neighbour Comparison — Custom vs Gensim Word2Vec\nIIT Jodhpur Corpus',
             color='white', fontsize=13, fontweight='bold', y=1.01)

model_keys = ['custom_sg', 'custom_cb', 'gensim_sg', 'gensim_cb']
for row, word in enumerate(QUERY_WORDS):
    for col, key in enumerate(model_keys):
        ax   = axes[row][col]
        nbrs = results_neighbors[word][key]
        ax_style(ax,
                 title=f'{LABELS[key]}' if row == 0 else '',
                 ylabel=f'"{word}"'     if col == 0 else '')
        if nbrs:
            ws = [n[0] for n in nbrs]
            ss = [n[1] for n in nbrs]
            bars = ax.barh(range(len(ws)), ss, color=COLORS[key], alpha=0.85, edgecolor='none')
            ax.set_yticks(range(len(ws)))
            ax.set_yticklabels(ws, color='white', fontsize=8)
            ax.set_xlim(0, 1)
            for b, score in zip(bars, ss):
                ax.text(b.get_width() + 0.01, b.get_y() + b.get_height()/2,
                        f'{score:.3f}', va='center', color='#ccc', fontsize=7.5)
        else:
            ax.text(0.5, 0.5, 'Not in vocab', ha='center', va='center',
                    color='#888', transform=ax.transAxes, fontsize=9)

plt.tight_layout()
plt.savefig('./outputs/comparison_neighbors.png', dpi=130,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("    Saved: comparison_neighbors.png")

# ─────────────────────────────────────────────────────────────
# PLOT 2 — COSINE SIMILARITY HEATMAPS SIDE BY SIDE
# ─────────────────────────────────────────────────────────────
HEATMAP_WORDS = [w for w in ['research','phd','student','exam','btech','mtech',
                               'faculty','department','thesis','course','semester',
                               'professor','lab','grade','program']
                 if w in vocab.word2idx]

def build_sim_matrix_custom(model, words):
    n = len(words)
    M = np.zeros((n, n))
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            s = model.cosine_similarity(w1, w2)
            M[i, j] = s if s is not None else 0.0
    return M

def build_sim_matrix_gensim(model, words):
    n = len(words)
    M = np.zeros((n, n))
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if w1 in model.wv and w2 in model.wv:
                M[i, j] = float(model.wv.similarity(w1, w2))
    return M

mats = {
    'Custom\nSkip-gram':  build_sim_matrix_custom(custom_sg, HEATMAP_WORDS),
    'Custom\nCBOW':       build_sim_matrix_custom(custom_cb, HEATMAP_WORDS),
    'Gensim\nSkip-gram':  build_sim_matrix_gensim(gensim_sg, HEATMAP_WORDS),
    'Gensim\nCBOW':       build_sim_matrix_gensim(gensim_cb, HEATMAP_WORDS),
}

fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor=DARK)
fig.patch.set_facecolor(DARK)
fig.suptitle('Cosine Similarity Matrices — Custom vs Gensim',
             color='white', fontsize=13, fontweight='bold')

for ax, (title, M) in zip(axes, mats.items()):
    ax.set_facecolor('#0d1117')
    n = len(HEATMAP_WORDS)
    im = ax.imshow(M, cmap='plasma', vmin=-0.1, vmax=1.0, aspect='auto')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(HEATMAP_WORDS, rotation=45, ha='right', color='white', fontsize=7)
    ax.set_yticklabels(HEATMAP_WORDS, color='white', fontsize=7)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center',
                    fontsize=5.5, color='white' if M[i,j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

plt.tight_layout()
plt.savefig('./outputs/comparison_heatmap.png', dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("    Saved: comparison_heatmap.png")

# ─────────────────────────────────────────────────────────────
# PLOT 3 — PCA COMPARISON
# ─────────────────────────────────────────────────────────────
word_groups = {
    'Academic':    (['btech','mtech','phd','undergraduate','postgraduate','degree','program','course'], '#00d4ff'),
    'Research':    (['research','thesis','publication','lab','experiment','paper'], '#ff6b35'),
    'People':      (['student','faculty','professor','scholar','supervisor','researcher'], '#00ff88'),
    'Examination': (['exam','grade','cgpa','semester','assessment','attendance'], '#ffcc02'),
    'Department':  (['department','engineering','chemistry','electrical','mechanical','civil'], '#a78bfa'),
}

def get_vecs_custom(model, groups):
    words, colors_list = [], []
    for grp, (ws, col) in groups.items():
        for w in ws:
            if w in model.vocab.word2idx:
                words.append(w); colors_list.append(col)
    vecs = np.array([model.W_in[model.vocab.word2idx[w]] for w in words])
    return words, colors_list, vecs

def get_vecs_gensim(model, groups):
    words, colors_list = [], []
    for grp, (ws, col) in groups.items():
        for w in ws:
            if w in model.wv:
                words.append(w); colors_list.append(col)
    vecs = np.array([model.wv[w] for w in words])
    return words, colors_list, vecs

fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor=DARK)
fig.patch.set_facecolor(DARK)
fig.suptitle('PCA 2D Projections — Custom vs Gensim Word2Vec',
             color='white', fontsize=13, fontweight='bold')

sources = [
    ('Custom Skip-gram',  *get_vecs_custom(custom_sg, word_groups)),
    ('Custom CBOW',       *get_vecs_custom(custom_cb, word_groups)),
    ('Gensim Skip-gram',  *get_vecs_gensim(gensim_sg, word_groups)),
    ('Gensim CBOW',       *get_vecs_gensim(gensim_cb, word_groups)),
]

for ax, (title, words, clrs, vecs) in zip(axes, sources):
    ax_style(ax, title=title, xlabel='PC 1', ylabel='PC 2')
    if len(vecs) >= 4:
        proj = pca_2d(vecs)
        for i, (word, col) in enumerate(zip(words, clrs)):
            ax.scatter(proj[i,0], proj[i,1], color=col, s=80, alpha=0.9,
                       edgecolors='white', linewidths=0.4, zorder=3)
            ax.annotate(word, (proj[i,0], proj[i,1]),
                        textcoords='offset points', xytext=(4,3),
                        fontsize=6.5, color='white', alpha=0.85)

legend_patches = [mpatches.Patch(color=v[1], label=k) for k,v in word_groups.items()]
axes[-1].legend(handles=legend_patches, loc='lower right',
                facecolor='#1f2937', edgecolor='#444',
                labelcolor='white', fontsize=7)

plt.tight_layout()
plt.savefig('./outputs/comparison_pca.png', dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("    Saved: comparison_pca.png")

# ─────────────────────────────────────────────────────────────
# PLOT 4 — SUMMARY METRICS BAR CHART
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK)
fig.patch.set_facecolor(DARK)
fig.suptitle('Quantitative Comparison — Custom vs Gensim Word2Vec',
             color='white', fontsize=13, fontweight='bold')

model_names  = ['Custom\nSkip-gram', 'Custom\nCBOW', 'Gensim\nSkip-gram', 'Gensim\nCBOW']
bar_colors   = [COLORS['custom_sg'], COLORS['custom_cb'],
                COLORS['gensim_sg'], COLORS['gensim_cb']]
x            = np.arange(4)
w            = 0.55

# Subplot 1: Similarity MSE (lower = better)
ax = axes[0]
ax_style(ax, title='Similarity MSE  (lower = better)', ylabel='MSE')
vals = [mse_custom_sg, mse_custom_cb, mse_gensim_sg, mse_gensim_cb]
bars = ax.bar(x, vals, width=w, color=bar_colors, alpha=0.85, edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(model_names, color='white', fontsize=8)
best_idx = int(np.argmin(vals))
for i, (b, v) in enumerate(zip(bars, vals)):
    label = f'{v:.4f}' + (' ★' if i == best_idx else '')
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
            label, ha='center', va='bottom', color='white', fontsize=8,
            fontweight='bold' if i == best_idx else 'normal')

# Subplot 2: Analogy Top-5 Accuracy (higher = better)
ax = axes[1]
ax_style(ax, title='Analogy Top-5 Accuracy  (higher = better)', ylabel='Accuracy')
vals = [acc_custom_sg, acc_custom_cb, acc_gensim_sg, acc_gensim_cb]
bars = ax.bar(x, vals, width=w, color=bar_colors, alpha=0.85, edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(model_names, color='white', fontsize=8)
ax.set_ylim(0, 1.15)
best_idx = int(np.argmax(vals))
for i, (b, v) in enumerate(zip(bars, vals)):
    label = f'{v:.2f}' + (' ★' if i == best_idx else '')
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            label, ha='center', va='bottom', color='white', fontsize=8,
            fontweight='bold' if i == best_idx else 'normal')

# Subplot 3: Training Time
ax = axes[2]
ax_style(ax, title='Training Time (seconds)', ylabel='Seconds')
times = [custom_sg_time, custom_cb_time, gensim_sg_time, gensim_cb_time]
bars  = ax.bar(x, times, width=w, color=bar_colors, alpha=0.85, edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(model_names, color='white', fontsize=8)
for b, t in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
            f'{t:.1f}s', ha='center', va='bottom', color='white', fontsize=8)

plt.tight_layout()
plt.savefig('./outputs/comparison_metrics.png', dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("    Saved: comparison_metrics.png")

# ─────────────────────────────────────────────────────────────
# PLOT 5 — ANALOGY DETAIL TABLE
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 5), facecolor=DARK)
fig.patch.set_facecolor(DARK)
ax.axis('off')
ax.set_title('Analogy Experiment Results — All 4 Models', color='white',
             fontsize=12, fontweight='bold', pad=10)

col_headers = ['Analogy', 'Expected', 'Custom SG Top-3', 'Custom CBOW Top-3',
               'Gensim SG Top-3', 'Gensim CBOW Top-3']
table_data  = [col_headers]
for i, (a, b, c, expected, desc) in enumerate(ANALOGY_TESTS):
    csg = ', '.join(det_custom_sg[i]['got'][:3]) if det_custom_sg[i]['got'] else '—'
    ccb = ', '.join(det_custom_cb[i]['got'][:3]) if det_custom_cb[i]['got'] else '—'
    gsg = ', '.join(det_gensim_sg[i]['got'][:3]) if det_gensim_sg[i]['got'] else '—'
    gcb = ', '.join(det_gensim_cb[i]['got'][:3]) if det_gensim_cb[i]['got'] else '—'
    table_data.append([desc, expected, csg, ccb, gsg, gcb])

tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.8)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#0a2342'); cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#111827' if r % 2 else '#1a2535')
        cell.set_text_props(color='white')
    cell.set_edgecolor('#333')

plt.tight_layout()
plt.savefig('./outputs/comparison_analogy.png', dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("    Saved: comparison_analogy.png")

# ─────────────────────────────────────────────────────────────
# SAVE COMPARISON RESULTS JSON
# ─────────────────────────────────────────────────────────────
comparison_results = {
    'hyperparams': {
        'dim': BEST_DIM, 'window': BEST_WINDOW,
        'neg': BEST_NEG, 'epochs': EPOCHS,
    },
    'training_time_seconds': {
        'custom_sg': round(custom_sg_time, 2),
        'custom_cb': round(custom_cb_time, 2),
        'gensim_sg': round(gensim_sg_time, 2),
        'gensim_cb': round(gensim_cb_time, 2),
    },
    'similarity_mse': {
        'custom_sg': round(mse_custom_sg, 4),
        'custom_cb': round(mse_custom_cb, 4),
        'gensim_sg': round(mse_gensim_sg, 4),
        'gensim_cb': round(mse_gensim_cb, 4),
    },
    'analogy_top5_accuracy': {
        'custom_sg': round(acc_custom_sg, 4),
        'custom_cb': round(acc_custom_cb, 4),
        'gensim_sg': round(acc_gensim_sg, 4),
        'gensim_cb': round(acc_gensim_cb, 4),
    },
    'analogy_details': {
        'custom_sg': det_custom_sg,
        'custom_cb': det_custom_cb,
        'gensim_sg': det_gensim_sg,
        'gensim_cb': det_gensim_cb,
    },
    'neighbors': {
        word: {
            key: [(n, round(s, 4)) for n, s in nbrs]
            for key, nbrs in results_neighbors[word].items()
        }
        for word in QUERY_WORDS
    },
}

with open('./outputs/comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)
print("    Saved: comparison_results.json")

# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<22} {'Sim MSE':>10} {'Analogy Acc':>13} {'Train Time':>12}")
print("-"*60)
for name, mse, acc, tt in [
    ('Custom Skip-gram',  mse_custom_sg, acc_custom_sg, custom_sg_time),
    ('Custom CBOW',       mse_custom_cb, acc_custom_cb, custom_cb_time),
    ('Gensim Skip-gram',  mse_gensim_sg, acc_gensim_sg, gensim_sg_time),
    ('Gensim CBOW',       mse_gensim_cb, acc_gensim_cb, gensim_cb_time),
]:
    print(f"  {name:<20} {mse:>10.4f} {acc:>13.2f} {tt:>10.1f}s")
print("="*60)
print("\nAll outputs saved to ./outputs/")
print("Next step: run generate_report_v2.py to build the updated PDF report.")