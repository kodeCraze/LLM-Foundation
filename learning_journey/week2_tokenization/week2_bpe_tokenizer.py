# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Week 2: Build a BPE Tokenizer From Scratch.

This script implements Byte Pair Encoding tokenization from scratch.
It's the foundation of modern LLMs (GPT, Llama, etc. all use BPE variants).

BPE Intuition:
- Start with individual characters
- Find most frequent adjacent pairs
- Merge them into new tokens
- Repeat until vocab size is reached
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import json


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation from scratch.
    
    This is a simplified but fully functional BPE tokenizer that demonstrates
    the core algorithm used in GPT-2, GPT-3, Llama, etc.
    """
    
    def __init__(self, vocab_size: int = 100):
        """Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (includes base characters)
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}  # token -> id
        self.inverse_vocab: Dict[int, str] = {}  # id -> token
        self.merges: List[Tuple[str, str]] = []  # list of (a, b) merges
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+""")
        
    def _preprocess(self, text: str) -> str:
        """Basic preprocessing."""
        return text.strip().lower()
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Split text into initial word tokens (pre-tokenization)."""
        # GPT-2 style pre-tokenization
        matches = self.pattern.findall(self._preprocess(text))
        return [match.strip() for match in matches if match.strip()]
    
    def _get_char_tokens(self, word: str) -> List[str]:
        """Convert word to character tokens with end-of-word marker."""
        # Add </w> marker to indicate word boundary
        return list(word) + ['</w>']
    
    def _get_pair_frequencies(self, word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of all adjacent token pairs."""
        pairs = defaultdict(int)
        for word_tokens, freq in word_freqs.items():
            for i in range(len(word_tokens) - 1):
                pairs[(word_tokens[i], word_tokens[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Apply merge to all words in vocabulary."""
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        new_word_freqs = {}
        for word_tokens in word_freqs:
            word_str = ' '.join(word_tokens)
            new_word_str = pattern.sub(''.join(pair), word_str)
            new_word_tokens = tuple(new_word_str.split())
            new_word_freqs[new_word_tokens] = word_freqs[word_tokens]
        
        return new_word_freqs
    
    def train(self, texts: List[str]) -> None:
        """Train BPE tokenizer on corpus.
        
        Args:
            texts: List of training texts
        """
        print("=" * 70)
        print("🔤 TRAINING BPE TOKENIZER")
        print("=" * 70)
        print()
        
        # Step 1: Pre-tokenization and word frequency counting
        print("Step 1: Building word frequency table...")
        word_freqs = defaultdict(int)
        for text in texts:
            for word in self._get_word_tokens(text):
                word_freqs[word] += 1
        
        print(f"  Found {len(word_freqs)} unique words")
        print(f"  Examples: {list(word_freqs.keys())[:10]}")
        print()
        
        # Step 2: Convert to character tokens
        print("Step 2: Converting to character-level tokens...")
        word_token_freqs = {}
        for word, freq in word_freqs.items():
            word_token_freqs[tuple(self._get_char_tokens(word))] = freq
        
        # Step 3: Build initial vocabulary from characters
        print("Step 3: Building initial character vocabulary...")
        all_chars = set()
        for word_tokens in word_token_freqs:
            for char in word_tokens:
                all_chars.add(char)
        
        self.vocab = {char: i for i, char in enumerate(sorted(all_chars))}
        print(f"  Initial vocab size: {len(self.vocab)} (characters + </w>)")
        print(f"  Characters: {sorted(all_chars)[:20]}{'...' if len(all_chars) > 20 else ''}")
        print()
        
        # Step 4: BPE Merges
        print(f"Step 4: Performing BPE merges (target: {self.vocab_size} tokens)...")
        print()
        
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # Find most frequent pair
            pairs = self._get_pair_frequencies(word_token_freqs)
            
            if not pairs:
                print("  No more pairs to merge!")
                break
            
            best_pair = max(pairs, key=pairs.get)
            word_token_freqs = self._merge_vocab(best_pair, word_token_freqs)
            
            # Add merged token to vocab
            merged_token = ''.join(best_pair)
            self.merges.append(best_pair)
            self.vocab[merged_token] = len(self.vocab)
            
            if (i + 1) % 10 == 0 or i < 5:
                print(f"  Merge {i+1}: {best_pair[0]} + {best_pair[1]} → {merged_token} "
                      f"(frequency: {pairs[best_pair]})")
        
        print()
        print(f"✅ Training complete! Final vocab size: {len(self.vocab)}")
        print(f"   Total merges: {len(self.merges)}")
        print()
        
        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        tokens = []
        for word in self._get_word_tokens(text):
            word_tokens = self._get_char_tokens(word)
            
            # Apply merges in order
            for merge in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and \
                       word_tokens[i] == merge[0] and \
                       word_tokens[i + 1] == merge[1]:
                        new_tokens.append(''.join(merge))
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Unknown token - fall back to characters
                    for char in token.replace('</w>', ''):
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Reconstructed text
        """
        tokens = [self.inverse_vocab.get(id, '<unk>') for id in token_ids]
        text = ''.join(tokens)
        # Remove end-of-word markers and clean up
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def visualize_tokenization(self, text: str) -> None:
        """Visualize how a specific text gets tokenized."""
        print("=" * 70)
        print(f"🔍 TOKENIZATION VISUALIZATION")
        print(f"   Text: '{text}'")
        print("=" * 70)
        print()
        
        # Step-by-step
        print("Pre-tokenization (word splitting):")
        words = self._get_word_tokens(text)
        print(f"  {words}")
        print()
        
        print("Character-level breakdown:")
        for word in words:
            chars = self._get_char_tokens(word)
            print(f"  '{word}' → {chars}")
        print()
        
        print("After BPE merging:")
        token_ids = self.encode(text)
        tokens = [self.inverse_vocab[id] for id in token_ids]
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print()
        
        print("Decoded back:")
        decoded = self.decode(token_ids)
        print(f"  '{decoded}'")
        print()
        
        # Statistics
        print("Statistics:")
        print(f"  Characters: {len(text)}")
        print(f"  Tokens: {len(token_ids)}")
        print(f"  Compression ratio: {len(text) / len(token_ids):.2f} chars/token")
        print()
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str) -> None:
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = {k: int(v) for k, v in data['vocab'].items()}
        self.merges = [tuple(m) for m in data['merges']]
        self.vocab_size = data['vocab_size']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tokenizer loaded from {path}")


def get_sample_corpus() -> List[str]:
    """Returns a small sample corpus for training."""
    return [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat and a dog played",
        "the mat was soft and warm",
        "the log was hard and rough",
        "cats and dogs are friends",
        "the cat sleeps on the mat",
        "the dog sleeps on the log",
        "a warm soft mat for the cat",
        "a rough hard log for the dog",
        "cats like soft mats",
        "dogs like rough logs",
        "the cat and dog sat together",
        "mats and logs are different",
        "the soft warm mat",
        "the hard rough log",
    ]


def compare_tokenizers() -> None:
    """Compares different tokenization approaches."""
    print("=" * 70)
    print("📊 TOKENIZER COMPARISON")
    print("=" * 70)
    print()
    
    test_text = "the cat sat on the mat"
    
    print(f"Test text: '{test_text}'")
    print()
    
    # Character-level
    print("1. CHARACTER-LEVEL TOKENIZATION:")
    print(f"   Tokens: {list(test_text)}")
    print(f"   Count: {len(test_text)}")
    print("   → Very simple, but no word meaning captured")
    print()
    
    # Word-level
    print("2. WORD-LEVEL TOKENIZATION:")
    words = test_text.split()
    print(f"   Tokens: {words}")
    print(f"   Count: {len(words)}")
    print("   → Captures meaning, but huge vocabulary")
    print("   → Can't handle typos or new words")
    print()
    
    # BPE (small vocab)
    print("3. BPE TOKENIZATION (vocab size 50):")
    corpus = get_sample_corpus()
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus)
    tokens = tokenizer.encode(test_text)
    print(f"   Tokens: {[tokenizer.inverse_vocab[t] for t in tokens]}")
    print(f"   Count: {len(tokens)}")
    print("   → Balance: subword units capture meaning + generalization")
    print()
    
    # BPE (larger vocab)
    print("4. BPE TOKENIZATION (vocab size 100):")
    tokenizer_large = BPETokenizer(vocab_size=100)
    tokenizer_large.train(corpus)
    tokens_large = tokenizer_large.encode(test_text)
    print(f"   Tokens: {[tokenizer_large.inverse_vocab[t] for t in tokens_large]}")
    print(f"   Count: {len(tokens_large)}")
    print("   → Larger vocab = more whole words, fewer splits")
    print()


def main():
    """Main execution for Week 2."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "WEEK 2: BPE TOKENIZATION" + " " * 24 + "║")
    print("║" + " " * 12 + "Building a Tokenizer From Scratch" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Get sample corpus
    corpus = get_sample_corpus()
    print("Training corpus (16 sentences):")
    for i, text in enumerate(corpus, 1):
        print(f"  {i:2}. {text}")
    print()
    
    # Train tokenizer
    vocab_size = 80
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus)
    
    # Show learned vocabulary
    print("=" * 70)
    print("📚 LEARNED VOCABULARY (Sample)")
    print("=" * 70)
    print()
    
    # Sort by length (longer = more merged)
    sorted_vocab = sorted(tokenizer.vocab.keys(), key=len, reverse=True)
    print("Longest tokens (most merged):")
    for token in sorted_vocab[:20]:
        if len(token) > 2:
            print(f"  '{token}' → ID {tokenizer.vocab[token]}")
    
    print()
    print("Character tokens (base vocabulary):")
    char_tokens = [t for t in sorted_vocab if len(t) == 1]
    print(f"  {char_tokens}")
    print()
    
    # Visualize tokenization
    test_texts = [
        "the cat sat on the mat",
        "cats and dogs are friends",
        "a warm soft mat",
        "the dog played on the log",
    ]
    
    for text in test_texts:
        tokenizer.visualize_tokenization(text)
    
    # Compare approaches
    compare_tokenizers()
    
    # Test encoding/decoding round-trip
    print("=" * 70)
    print("🔄 ENCODE → DECODE ROUND-TRIP TEST")
    print("=" * 70)
    print()
    
    test_sentences = [
        "the cat sleeps",
        "a dog played",
        "warm soft mats",
    ]
    
    for text in test_sentences:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: '{text}'")
        print(f"Encoded:  {encoded}")
        print(f"Decoded:  '{decoded}'")
        match = "✅" if text == decoded else "⚠️"
        print(f"Match:    {match}")
        print()
    
    # Save tokenizer
    tokenizer.save("week2_tokenizer.json")
    
    print("=" * 70)
    print("🎯 WEEK 2 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. BPE ALGORITHM:")
    print("   • Start: character-level tokens")
    print("   • Iteratively merge most frequent adjacent pairs")
    print("   • Result: vocabulary of subword units")
    print()
    print("2. WHY BPE WORKS:")
    print("   • Common words → single tokens ('the', 'cat')")
    print("   • Rare words → split into subwords ('playing' → 'play' + 'ing')")
    print("   • Infinite vocabulary from finite base characters")
    print()
    print("3. YOUR CODEBASE:")
    print("   • Check tokenization/ directory for your tokenizers")
    print("   • Compare with this implementation!")
    print()
    print("4. MODERN LLMs:")
    print("   • GPT-2/3/4: BPE with ~50K vocab")
    print("   • Llama: BPE variant with special tokens")
    print("   • Gemma: SentencePiece (similar subword approach)")
    print()
    print("=" * 70)
    print()
    print("Next: Explore your tokenization/ directory and compare!")
    print("      Or run this with a larger corpus to see improvements.")
    print()


if __name__ == "__main__":
    main()
