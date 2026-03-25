"""
TASK 1: DATASET PREPARATION
IIT Jodhpur Word2Vec Assignment
Preprocessing pipeline for collected corpus

This module handles all text cleaning and tokenization steps
before feeding data into the Word2Vec model.
"""

import re
import string
import collections
import sys

# Insert current working directory into Python path so local modules can be imported
sys.path.insert(0, 'os.getcwd()')

# Import the raw corpus data from the local corpus file
from iitj_corpus_raw import get_corpus

# ─── STOP WORDS ────────────────────────────────────────────────────────────────
# A set of common English words that carry little semantic meaning.
# These are used optionally during preprocessing — kept here for reference but
# Word2Vec training benefits from retaining stop words (the model learns context
# from high-frequency words too). They are only filtered during the word cloud.
STOP_WORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','up','about','into','over','after','is','are','was','were',
    'be','been','being','have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','that','this','these','those',
    'i','we','you','he','she','it','they','me','us','him','her','them',
    'my','our','your','his','its','their','what','which','who','when','where',
    'how','all','each','every','both','few','more','most','other','some',
    'such','no','not','only','same','so','than','too','very','just','as',
    'if','then','because','while','although','since','before','during',
    'through','between','among','under','above','across','against','along',
    'also','well','can','per','any','been','there'
}


def preprocess_text(text):
    """
    Full preprocessing pipeline applied to a single document string.
    Steps:
      1. Remove section numbers (e.g. "1.2", "2.1.3")
      2. Remove standalone integers
      3. Strip URLs and email addresses
      4. Remove non-English characters except basic punctuation
      5. Normalize whitespace
      6. Lowercase the entire text
      7. Split into sentences on '.', '!', '?'
      8. Tokenize each sentence by whitespace
      9. Strip punctuation from token edges and keep alphabetic tokens >= 2 chars
     10. Discard sentences with fewer than 3 tokens (too short to be useful)

    Args:
        text (str): Raw document text.

    Returns:
        list of list of str: A list of sentences, where each sentence
                             is a list of cleaned string tokens.
    """
    # Remove section numbers like "1.2", "2.1.3" that appear in academic PDFs
    text = re.sub(r'\b\d+\.\d+(\.\d+)?\b', '', text)

    # Remove bare integers (page numbers, years used as standalone digits, etc.)
    text = re.sub(r'\b\d+\b', '', text)

    # Remove hyperlinks starting with http/https or www
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove email addresses (pattern: anything@anything)
    text = re.sub(r'\S+@\S+', '', text)

    # Keep only letters, whitespace, and sentence-ending punctuation
    # All other characters (brackets, slashes, special symbols) → space
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', ' ', text)

    # Collapse multiple consecutive spaces/tabs/newlines into a single space
    text = re.sub(r'\s+', ' ', text)

    # Lowercase so "Research" and "research" map to the same token
    text = text.lower()

    # Split the cleaned text into raw sentence fragments on sentence-ending punctuation
    sentence_endings = re.compile(r'[.!?]+')
    raw_sentences = sentence_endings.split(text)

    sentences = []
    for sent in raw_sentences:
        # Split on whitespace to get initial token list
        tokens = sent.split()

        # Strip any remaining punctuation characters from the start/end of each token
        # e.g. ",research." → "research"
        tokens = [t.strip(string.punctuation) for t in tokens]

        # Keep only purely alphabetic tokens of length >= 2
        # This removes single characters ('s', 'a') and numeric remnants
        tokens = [t for t in tokens if t.isalpha() and len(t) >= 2]

        # Skip very short sentences (less than 3 tokens) — not useful for context windows
        if len(tokens) >= 3:
            sentences.append(tokens)

    return sentences


def build_corpus(documents):
    """
    Apply preprocessing to all documents and return a flat list of sentences.

    Args:
        documents (list of str): Raw text documents.

    Returns:
        list of list of str: All sentences from all documents combined.
    """
    all_sentences = []
    for doc in documents:
        # Preprocess each document and extend the master sentence list
        sentences = preprocess_text(doc)
        all_sentences.extend(sentences)
    return all_sentences


def compute_stats(sentences, documents):
    """
    Compute basic corpus statistics useful for reporting and debugging.

    Args:
        sentences (list of list of str): Preprocessed sentences.
        documents (list of str): Original raw documents (for document count).

    Returns:
        dict: A dictionary containing:
            - 'num_documents': total number of source documents
            - 'num_sentences': total number of preprocessed sentences
            - 'total_tokens' : total token count (with repetition)
            - 'vocab_size'   : number of unique token types
            - 'top_50'       : list of (word, count) for the 50 most frequent words
            - 'freq'         : full Counter object of all token frequencies
    """
    # Flatten all sentences into one big token list for frequency counting
    all_tokens = [tok for sent in sentences for tok in sent]

    vocab = set(all_tokens)           # Unique word types
    freq = collections.Counter(all_tokens)  # Token → occurrence count

    stats = {
        'num_documents': len(documents),
        'num_sentences': len(sentences),
        'total_tokens': len(all_tokens),
        'vocab_size': len(vocab),
        'top_50': freq.most_common(50),   # Top 50 most frequent words
        'freq': freq                       # Full frequency distribution
    }
    return stats


def save_cleaned_corpus(sentences, path):
    """
    Write the cleaned corpus to a plain-text file, one sentence per line.
    The file includes a header with basic statistics.

    Args:
        sentences (list of list of str): Preprocessed tokenized sentences.
        path (str): Output file path (e.g. './outputs/corpus.txt').
    """
    with open(path, 'w', encoding='utf-8') as f:
        # Write a brief header so the file is self-documenting
        f.write("# CLEANED CORPUS — IIT JODHPUR WORD2VEC ASSIGNMENT\n")
        f.write(f"# Total sentences: {len(sentences)}\n")
        f.write(f"# Total tokens: {sum(len(s) for s in sentences)}\n\n")

        # Each sentence is written as space-separated tokens on its own line
        for sent in sentences:
            f.write(' '.join(sent) + '\n')

    print(f"Cleaned corpus saved to {path}")


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load raw documents from the corpus module
    documents = get_corpus()

    # Run the full preprocessing pipeline on all documents
    sentences = build_corpus(documents)

    # Compute and display descriptive statistics
    stats = compute_stats(sentences, documents)

    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Number of documents  : {stats['num_documents']}")
    print(f"Number of sentences  : {stats['num_sentences']}")
    print(f"Total tokens         : {stats['total_tokens']}")
    print(f"Vocabulary size      : {stats['vocab_size']}")
    print()
    print("Top 30 most frequent words:")
    for word, count in stats['top_50'][:30]:
        # Left-align each word in a 20-character field for tidy output
        print(f"  {word:<20} {count}")

    # Persist the cleaned corpus to disk for use by downstream tasks
    save_cleaned_corpus(sentences, './cleaned_corpus.txt')