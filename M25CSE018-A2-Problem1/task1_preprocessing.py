"""
TASK 1: DATASET PREPARATION
IIT Jodhpur Word2Vec Assignment
Preprocessing pipeline for collected corpus
"""

import re
import string
import collections
import sys
sys.path.insert(0, 'os.getcwd()')
from iitj_corpus_raw import get_corpus

# ─── STOP WORDS ────────────────────────────────────────────────────────────────
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
    Full preprocessing pipeline:
    1. Remove boilerplate / formatting artifacts
    2. Lowercase
    3. Remove non-English characters
    4. Tokenize
    5. Remove excessive punctuation
    6. Remove stop words (optional — kept for Word2Vec training)
    Returns list of sentences, each sentence is list of tokens
    """
    # Remove section numbers like "1.2", "2.1.3"
    text = re.sub(r'\b\d+\.\d+(\.\d+)?\b', '', text)
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters except sentence-ending punctuation
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Lowercase
    text = text.lower()
    # Split into sentences on . ! ?
    sentence_endings = re.compile(r'[.!?]+')
    raw_sentences = sentence_endings.split(text)
    
    sentences = []
    for sent in raw_sentences:
        # Tokenize by whitespace
        tokens = sent.split()
        # Remove purely punctuation tokens
        tokens = [t.strip(string.punctuation) for t in tokens]
        # Keep only alphabetic tokens of length >= 2
        tokens = [t for t in tokens if t.isalpha() and len(t) >= 2]
        if len(tokens) >= 3:  # Discard very short sentences
            sentences.append(tokens)
    return sentences

def build_corpus(documents):
    """Process all documents and return sentences + stats"""
    all_sentences = []
    for doc in documents:
        sentences = preprocess_text(doc)
        all_sentences.extend(sentences)
    return all_sentences

def compute_stats(sentences, documents):
    """Compute and print dataset statistics"""
    all_tokens = [tok for sent in sentences for tok in sent]
    vocab = set(all_tokens)
    freq = collections.Counter(all_tokens)
    
    stats = {
        'num_documents': len(documents),
        'num_sentences': len(sentences),
        'total_tokens': len(all_tokens),
        'vocab_size': len(vocab),
        'top_50': freq.most_common(50),
        'freq': freq
    }
    return stats

def save_cleaned_corpus(sentences, path):
    """Save cleaned corpus to file"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# CLEANED CORPUS — IIT JODHPUR WORD2VEC ASSIGNMENT\n")
        f.write(f"# Total sentences: {len(sentences)}\n")
        f.write(f"# Total tokens: {sum(len(s) for s in sentences)}\n\n")
        for sent in sentences:
            f.write(' '.join(sent) + '\n')
    print(f"Cleaned corpus saved to {path}")

if __name__ == '__main__':
    documents = get_corpus()
    sentences = build_corpus(documents)
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
        print(f"  {word:<20} {count}")
    
    save_cleaned_corpus(sentences, './cleaned_corpus.txt')