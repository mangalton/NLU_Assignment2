# NLU_Assignment2
# Word2Vec from Scratch — IIT Jodhpur NLP Assignment 2

**Student:** M25CSE018  
**Assignment:** A2 — Problem 1  
**Course:** Natural Language Processing, IIT Jodhpur

---

## Overview

This project implements Word2Vec entirely from scratch using NumPy, trained on a custom IIT Jodhpur corpus. Both **CBOW** (Continuous Bag of Words) and **Skip-gram** architectures are built with **Negative Sampling**, and the custom implementation is benchmarked against Gensim's Word2Vec.

---

## Project Structure

```
M25CSE018-A2-Problem1/
│
├── iitj_corpus_raw.py          # Raw corpus data (39 IIT Jodhpur documents)
├── task1_preprocessing.py      # Text preprocessing pipeline
├── task2_word2vec.py           # Word2Vec model (CBOW + Skip-gram) from scratch
├── comparision.py              # Task 5: Custom vs Gensim comparison
├── run_all.py                  # Master runner — executes the full pipeline
├── output.log                  # Training logs from the last full run
│
└── outputs/
    ├── corpus.txt                      # Cleaned, tokenised corpus
    ├── embedding_300d.txt              # Sample word vectors (300-dim, plain text)
    ├── results.json                    # Hyperparameter experiment results
    ├── comparison_results.json         # Custom vs Gensim numeric results
    ├── wordcloud.png                   # Frequency-based word cloud
    ├── word_frequency.png              # Top word frequency bar chart
    ├── training_loss.png               # Loss curves across epochs
    ├── hyperparam_comparison.png       # Hyperparameter sweep comparison
    ├── pca_cbow.png                    # PCA projection — CBOW embeddings
    ├── pca_skip_gram.png               # PCA projection — Skip-gram embeddings
    ├── tsne_cbow.png                   # t-SNE projection — CBOW embeddings
    ├── tsne_skip_gram.png              # t-SNE projection — Skip-gram embeddings
    ├── heatmap_cbow.png                # Cosine similarity heatmap — CBOW
    ├── heatmap_skip_gram.png           # Cosine similarity heatmap — Skip-gram
    ├── comparison_pca.png              # Side-by-side PCA: Custom vs Gensim
    ├── comparison_heatmap.png          # Side-by-side cosine heatmaps
    ├── comparison_heatmaps.png         # All-model heatmap grid
    ├── comparison_mse_bar.png          # MSE bar chart: Custom vs Gensim
    ├── comparison_metrics.png          # Full metrics comparison chart
    ├── comparison_analogy.png          # Analogy accuracy comparison
    ├── comparison_neighbors.png        # Nearest-neighbour comparison
    └── comparison_neighbor_overlap.png # Neighbour overlap between models
```

---

## Tasks

### Task 1 — Dataset Preparation (`task1_preprocessing.py`)
- Loads 39 domain-specific IIT Jodhpur documents via `iitj_corpus_raw.py`
- Preprocessing pipeline: URL/email removal → lowercase → tokenisation → stop-word filtering → sentence segmentation
- Keeps only alphabetic tokens of length ≥ 2 and sentences with ≥ 3 tokens
- Outputs corpus statistics and saves cleaned text to `outputs/corpus.txt`

**Corpus stats (from last run):**

| Metric | Value |
|--------|-------|
| Documents | 39 |
| Sentences | 1,866 |
| Total tokens | 38,707 |
| Vocabulary size | 4,903 |

---

### Task 2 — Word2Vec Training (`task2_word2vec.py`)
Pure NumPy implementation of Word2Vec with Negative Sampling. No deep learning framework is used.

**Key classes:**

- `Vocabulary` — builds word↔index mapping, filters by `min_count`, generates unigram table (raised to ¾ power) for negative sampling
- `Word2Vec` — unified model supporting both architectures:
  - `skipgram` — predicts context words from a center word
  - `cbow` — predicts center word from averaged context vectors
  - Linear learning rate decay across epochs
  - Dynamic window size (sampled uniformly each step)

**Hyperparameter configurations trained:**

| Label | Arch | Dim | Window | Neg Samples |
|-------|------|-----|--------|-------------|
| SG-100d-w5-n5 | Skip-gram | 100 | 5 | 5 |
| SG-50d-w3-n5 | Skip-gram | 50 | 3 | 5 |
| SG-100d-w7-n10 | Skip-gram | 100 | 7 | 10 |
| CBOW-100d-w5-n5 | CBOW | 100 | 5 | 5 |
| CBOW-50d-w3-n5 | CBOW | 50 | 3 | 5 |
| CBOW-100d-w7-n10 | CBOW | 100 | 7 | 10 |

---

### Task 5 — Framework Comparison (`comparision.py`)
Benchmarks the custom implementation against Gensim Word2Vec using identical hyperparameters (`dim=100`, `window=5`, `neg=2`, `epochs=30`).

**Results summary:**

| Model | Training Time | Similarity MSE | Analogy Top-5 Acc |
|-------|--------------|----------------|-------------------|
| Custom Skip-gram | 66.9s | 0.133 | 40% |
| Custom CBOW | 24.8s | 0.046 | 0% |
| Gensim Skip-gram | 1.03s | 0.102 | 60% |
| Gensim CBOW | 0.41s | 0.117 | 40% |

Notable: Custom CBOW achieves the **lowest similarity MSE** among all four models, while Gensim is ~65× faster due to optimised C internals.

---

## Requirements

```
numpy
matplotlib
gensim          # only required for comparision.py
```

Install dependencies:
```bash
pip install numpy matplotlib gensim
```

---

## Running the Code

### Full pipeline (all tasks + all plots)
```bash
python run_all.py
```

### Individual tasks
```bash
# Task 1: Preprocessing only
python task1_preprocessing.py

# Task 2: Train all Word2Vec models
python task2_word2vec.py

# Task 5: Custom vs Gensim comparison
python comparision.py
```

All outputs (plots, logs, model files, JSON results) are saved to `./outputs/`.

---
# Character-Level Name Generation — IIT Jodhpur NLP Assignment 2

**Student:** M25CSE018  
**Assignment:** A2 — Problem 2  
**Course:** Natural Language Processing, IIT Jodhpur

---

## Overview

This project implements three character-level language models from scratch using NumPy (no deep learning frameworks) for the task of **generative name synthesis**. All models are trained on a dataset of 948 Indian names and evaluated on novelty, diversity, and training loss reduction.

---

## Project Structure

```
M25CSE018-A2-Problem2/
│
├── training_names.txt          # 948 Indian names used for training
├── vanilla_rnn.py              # Task 1a: Single-layer Vanilla RNN
├── rnn_attention.py            # Task 1b: RNN with Bahdanau Attention
├── bilstm.py                   # Task 1c: Bidirectional LSTM (dual-head)
├── train_all.py                # Master trainer — runs all three models
├── evaluate.py                 # Task 2: Quantitative evaluation + plots
├── output.log                  # Training logs from last full run
├── evaluation.log              # Evaluation results from last full run
│
└── outputs/
    ├── vanilla_rnn.npz             # Saved Vanilla RNN weights
    ├── vanilla_rnn_names.txt       # 200 names generated by Vanilla RNN
    ├── rnn_attention_names.txt     # 200 names generated by RNN+Attention
    ├── blstm_names.txt             # 199 names generated by BLSTM
    ├── training_losses.csv         # Per-epoch loss for all 3 models (150 epochs)
    └── loss_curves.png             # Training loss visualisation plot
```

---

## Models

### 1. Vanilla RNN (`vanilla_rnn.py`)

A single-layer recurrent neural network for character-level sequence modelling.

**Architecture:** `Input (one-hot) → Wxh·x + Whh·h_prev + bh → tanh → Why·h + by → Softmax`

| Hyperparameter | Value |
|----------------|-------|
| Hidden size | 128 |
| Layers | 1 |
| Learning rate | 0.005 (Adagrad) |
| Gradient clip | 5.0 |
| Epochs | 150 |
| Trainable params | 23,708 |


### 2. RNN with Attention (`rnn_attention.py`)

An encoder-decoder RNN with additive (Bahdanau-style) attention.

**Architecture:** Encoder RNN reads the character prefix → Attention computes a weighted context over encoder hidden states → Decoder RNN receives `[x_t ; context]` concatenated as input → Softmax output

| Hyperparameter | Value |
|----------------|-------|
| Hidden size (encoder & decoder) | 128 |
| Attention projection size | 64 |
| Learning rate | 0.005 (Adagrad) |
| Gradient clip | 5.0 |
| Epochs | 150 |
| Trainable params | 76,700 |


### 3. Bidirectional LSTM (`bilstm.py`)

A BiLSTM with a **dual output head** design that resolves the train/generate distribution mismatch.

**Architecture (training):** `Fwd-LSTM(96) + Bwd-LSTM(96) → concat [h_fwd || h_bwd] → Combined head (192→V)`  
**Architecture (generation):** `Fwd-LSTM(96) → Forward-only head (96→V)` *(no backward LSTM used)*

| Hyperparameter | Value |
|----------------|-------|
| Hidden size per direction | 96 |
| Combined hidden size | 192 |
| Auxiliary loss weight | 0.5 |
| Learning rate | 0.003 (Adagrad) |
| Gradient clip | 5.0 |
| Epochs | 150 |
| Trainable params | 104,120 |



## Training Results

All three models trained for 150 epochs on 948 names.

| Model | Params | Initial Loss | Final Loss | Loss Reduction | Training Time |
|-------|--------|-------------|------------|----------------|---------------|
| Vanilla RNN | 23,708 | 18.82 | 13.08 | 30.5% | 67.0s |
| BLSTM | 104,120 | 22.89 | 7.02 | **69.3%** | 273.1s |
| RNN+Attention | 76,700 | 18.21 | 10.67 | 41.4% | 308.5s |

---

## Evaluation Results (Task 2)

Each model generated 200 names, evaluated against the 948-name training set.

| Model | Generated | Novelty | Diversity | Valid Length | Avg Len |
|-------|-----------|---------|-----------|--------------|---------|
| Vanilla RNN | 200 | 97.5% | 98.5% | 100% | 6.15 |
| BLSTM | 199 | 97.0% | **99.5%** | 100% | 6.10 |
| RNN+Attention | 200 | 86.0% | 98.0% | 100% | 6.19 |


---

## Requirements

```
numpy
matplotlib
pandas
```

Install dependencies:
```bash
pip install numpy matplotlib pandas
```

---

## Running the Code

### Train all models and generate names
```bash
python train_all.py
```

### Run evaluation (requires outputs from train_all.py)
```bash
python evaluate.py
```
```

All generated names and loss curves are saved to `./outputs/`. The trained Vanilla RNN weights are saved as `outputs/vanilla_rnn.npz`.

---
