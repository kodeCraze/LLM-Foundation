# Week 2: Tokenization & Embeddings Deep Dive ✅

**Focus:** Understanding how text becomes numbers (the true foundation)

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand how BPE tokenization works (the algorithm behind GPT, Llama, etc.)
- ✅ Build a working tokenizer from scratch
- ✅ Visualize how words exist as points in embedding space
- ✅ See why "similar words have similar vectors"
- ✅ Connect to your existing `tokenization/` and `embeddings/` directories

---

## Week 2 Scripts

### 1. `week2_bpe_tokenizer.py`
**Purpose:** Build a complete BPE tokenizer from scratch

**What you'll learn:**
- The BPE algorithm step-by-step
- Pre-tokenization (word splitting)
- Character initialization
- Iterative merge process
- Encoding and decoding

**Run it:**
```bash
cd learning_journey/week2_tokenization
python week2_bpe_tokenizer.py
```

**Output:**
- Training process visualization
- Vocabulary inspection
- Tokenization examples
- Round-trip encode/decode tests
- Saved tokenizer: `week2_tokenizer.json`

---

### 2. `week2_embedding_visualization.py`
**Purpose:** Visualize embedding space with PCA and t-SNE

**What you'll learn:**
- How embeddings create semantic structure
- Cosine similarity between words
- Word analogies (king - man + woman ≈ queen)
- High-dimensional space visualization

**Run it:**
```bash
python week2_embedding_visualization.py
```

**Output:**
- `embedding_space.png` - 2D visualization of word relationships
- Similarity comparisons
- Analogy demonstrations

---

## Key Concepts

### BPE Tokenization

**Algorithm:**
1. Start with character-level vocabulary
2. Count frequency of adjacent pairs
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reached

**Why it works:**
- Common words → single tokens ("the", "and")
- Rare words → split into subwords ("playing" → "play" + "ing")
- Infinite vocabulary from finite base characters
- Handles typos and new words gracefully

**In your codebase:**
- Check `tokenization/` directory
- Compare with this implementation

### Word Embeddings

**Core idea:**
- Map discrete tokens to continuous vectors
- Similar words → similar vectors
- Relationships become vector arithmetic

**Visual analogy:**
```
High-dimensional space (simplified to 2D):

    cat ●──────────● dog
        │           │
        │   animal  │
        │   space   │
        │           │
   kitten ●──────────● puppy

Direction kitten→cat ≈ Direction puppy→dog
(young → adult)
```

**In your codebase:**
- `embeddings/` directory
- `layers.py` line 76: `layers.Embedding(...)`

---

## Week 2 Exercises

### Exercise 1: Train on Larger Corpus
1. Get Project Gutenberg text (free books)
2. Train tokenizer on 1000+ sentences
3. Observe how vocabulary changes

### Exercise 2: Compare Tokenizations
```python
# Test different texts
texts = [
    "hello world",
    "neural networks",
    "transformer architecture",
    "unseenword",
]

for text in texts:
    tokenizer.visualize_tokenization(text)
```

### Exercise 3: Explore Your Tokenization Directory
```bash
ls -la tokenization/
cat tokenization/__init__.py
# What tokenizers do you already have?
```

### Exercise 4: Vocabulary Analysis
```python
# After training
sorted_vocab = sorted(tokenizer.vocab.keys(), key=len, reverse=True)

# Questions to answer:
# 1. What are the 10 longest tokens?
# 2. What character has the lowest ID?
# 3. How many tokens are single characters?
```

---

## Connection to Modern LLMs

| Model | Tokenizer | Vocab Size | Notes |
|-------|-----------|------------|-------|
| GPT-2 | BPE | 50,257 | Original modern BPE |
| GPT-3/4 | BPE | 100,256+ | Extended |
| Llama 2/3 | BPE | 32,000 | Special tokens added |
| Gemma | SentencePiece | 256,000 | Different algorithm, same concept |
| BERT | WordPiece | 30,522 | Similar subword approach |

**Key insight:** All subword tokenization follows the same principle: break rare words, keep common words whole.

---

## Week 2 Completion Checklist

- [ ] Ran `week2_bpe_tokenizer.py` and understand output
- [ ] Can explain BPE algorithm in your own words
- [ ] Ran `week2_embedding_visualization.py` and saw clustering
- [ ] Understand why "cat" and "dog" have similar vectors
- [ ] Explored your `tokenization/` directory
- [ ] Completed at least 1 exercise
- [ ] Can answer: "Why not just use words as tokens?"

---

## Key Takeaways

### BPE Tokenization
1. **Starts with characters**, merges to subwords
2. **Frequency-driven**: most common pairs merge first
3. **Balance**: between character-level (too granular) and word-level (too sparse)
4. **Generalizes**: can tokenize any text, even unseen words

### Embeddings
1. **Discrete → Continuous**: tokens become vectors
2. **Semantic similarity**: similar words cluster in vector space
3. **Learned representations**: trained via backpropagation
4. **High-dimensional**: real models use 512-8192 dimensions

---

## Bridge to Week 3

Next week: **Training & The Forward Pass**
- How to train your tokenizer + embedding model
- Build a minimal training loop
- Understand loss functions
- Overfit to memorize a small dataset

---

## Resources

### Reading
- "The Illustrated BPE" (search for this blog post)
- "Neural Network Embeddings Explained"
- Your `tokenization/` directory code

### Interactive
- [Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) - Hugging Face tokenizer visualization
- Try different tokenizers on the same text

### Papers
- "Neural Machine Translation of Rare Words with Subword Units" (BPE paper)
- "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)

---

## Week 2 Status

**Your progress:**
- ✅ BPE tokenizer from scratch
- ✅ Embedding visualization
- ✅ Semantic similarity understanding
- ✅ Connection to your codebase

**Ready for Week 3?** 

If yes → See `../week3_training/`

If no → Re-run the scripts, modify parameters, experiment!

---

*Remember: Mastery > Speed. Take time to internalize these concepts.*
