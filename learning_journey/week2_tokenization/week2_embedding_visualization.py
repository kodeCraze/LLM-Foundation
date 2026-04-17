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

"""Week 2: Embedding Space Visualization.

This script demonstrates how token embeddings create semantic meaning in
high-dimensional space. It visualizes the concept that "similar words have
similar vectors" using dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple


class SimpleEmbeddingDemo:
    """Demonstrates embedding concepts with synthetic but meaningful data."""
    
    def __init__(self, vocab: List[str], embedding_dim: int = 16):
        """Initialize with vocabulary and embedding dimension.
        
        Args:
            vocab: List of words
            embedding_dim: Dimensionality of embeddings
        """
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        # Create synthetic embeddings with semantic structure
        self.embeddings = self._create_structured_embeddings()
        
    def _create_structured_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings with intentional semantic structure.
        
        We create embeddings where:
        - Similar words (cat, dog, pet) have similar vectors
        - Related words have specific directional relationships
        - The structure is visible in 2D projection
        """
        np.random.seed(42)  # For reproducibility
        embeddings = {}
        
        # Define semantic clusters
        clusters = {
            'animals': ['cat', 'dog', 'pet', 'animal', 'kitten', 'puppy'],
            'furniture': ['mat', 'bed', 'chair', 'table', 'furniture', 'sofa'],
            'actions': ['sit', 'sleep', 'run', 'play', 'walk', 'eat'],
            'descriptors': ['soft', 'warm', 'hard', 'rough', 'smooth', 'cold'],
        }
        
        # Base vectors for each cluster
        cluster_bases = {
            'animals': np.array([1.0, 0.5, 0.0, 0.0] + [0.0] * (self.embedding_dim - 4)),
            'furniture': np.array([0.0, 1.0, 0.5, 0.0] + [0.0] * (self.embedding_dim - 4)),
            'actions': np.array([0.5, 0.0, 1.0, 0.0] + [0.0] * (self.embedding_dim - 4)),
            'descriptors': np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * (self.embedding_dim - 4)),
        }
        
        # Add random noise to create distinct but related vectors
        for word in self.vocab:
            # Find which cluster this word belongs to
            cluster_name = None
            for cluster, words in clusters.items():
                if word in words:
                    cluster_name = cluster
                    break
            
            if cluster_name:
                # Base + noise
                base = cluster_bases[cluster_name].copy()
                noise = np.random.randn(self.embedding_dim) * 0.2
                embeddings[word] = base + noise
            else:
                # Random embedding for unknown words
                embeddings[word] = np.random.randn(self.embedding_dim) * 0.5
        
        return embeddings
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word."""
        return self.embeddings.get(word, np.random.randn(self.embedding_dim) * 0.5)
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words."""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        return dot_product / (norm1 * norm2)
    
    def analogy(self, a: str, b: str, c: str) -> str:
        """Solve analogy: a is to b as c is to ?
        
        Example: king - man + woman = queen
        """
        vec_a = self.get_embedding(a)
        vec_b = self.get_embedding(b)
        vec_c = self.get_embedding(c)
        
        # target = c + (b - a)
        target = vec_c + (vec_b - vec_a)
        
        # Find closest word
        best_word = None
        best_score = -float('inf')
        
        for word in self.vocab:
            if word in [a, b, c]:
                continue
            
            vec_word = self.get_embedding(word)
            score = np.dot(target, vec_word) / (np.linalg.norm(target) * np.linalg.norm(vec_word))
            
            if score > best_score:
                best_score = score
                best_word = word
        
        return best_word


def visualize_embedding_space(embedding_demo: SimpleEmbeddingDemo) -> None:
    """Visualize the embedding space using PCA and t-SNE."""
    print("=" * 70)
    print("📊 EMBEDDING SPACE VISUALIZATION")
    print("=" * 70)
    print()
    
    words = list(embedding_demo.embeddings.keys())
    vectors = np.array([embedding_demo.embeddings[w] for w in words])
    
    # Color by semantic category
    categories = {
        'animals': ['cat', 'dog', 'pet', 'animal', 'kitten', 'puppy'],
        'furniture': ['mat', 'bed', 'chair', 'table', 'furniture', 'sofa'],
        'actions': ['sit', 'sleep', 'run', 'play', 'walk', 'eat'],
        'descriptors': ['soft', 'warm', 'hard', 'rough', 'smooth', 'cold'],
    }
    
    colors_map = {
        'animals': 'red',
        'furniture': 'blue',
        'actions': 'green',
        'descriptors': 'orange',
        'other': 'gray'
    }
    
    def get_category(word):
        for cat, words in categories.items():
            if word in words:
                return cat
        return 'other'
    
    colors = [colors_map[get_category(w)] for w in words]
    
    # PCA projection
    pca = PCA(n_components=2)
    vectors_pca = pca.fit_transform(vectors)
    
    print("PCA Projection (captures most variance):")
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print()
    
    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    vectors_tsne = tsne.fit_transform(vectors)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA plot
    ax1 = axes[0]
    for category in colors_map.keys():
        mask = [get_category(w) == category for w in words]
        points = vectors_pca[mask]
        ax1.scatter(points[:, 0], points[:, 1], 
                   c=colors_map[category], label=category, s=100, alpha=0.7)
    
    # Add word labels
    for i, word in enumerate(words):
        ax1.annotate(word, (vectors_pca[i, 0], vectors_pca[i, 1]),
                    fontsize=9, ha='center', va='bottom')
    
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_title('Word Embeddings - PCA Projection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE plot
    ax2 = axes[1]
    for category in colors_map.keys():
        mask = [get_category(w) == category for w in words]
        points = vectors_tsne[mask]
        ax2.scatter(points[:, 0], points[:, 1], 
                   c=colors_map[category], label=category, s=100, alpha=0.7)
    
    # Add word labels
    for i, word in enumerate(words):
        ax2.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]),
                    fontsize=9, ha='center', va='bottom')
    
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title('Word Embeddings - t-SNE Projection\n(Local neighborhood preservation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_space.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to 'embedding_space.png'")
    print()


def demonstrate_similarity(embedding_demo: SimpleEmbeddingDemo) -> None:
    """Demonstrate cosine similarity between words."""
    print("=" * 70)
    print("📏 COSINE SIMILARITY DEMONSTRATION")
    print("=" * 70)
    print()
    
    print("Cosine similarity ranges from -1 (opposite) to 1 (identical)")
    print()
    
    # Similar pairs
    similar_pairs = [
        ('cat', 'dog'),
        ('cat', 'kitten'),
        ('mat', 'bed'),
        ('soft', 'warm'),
        ('sit', 'sleep'),
    ]
    
    print("Similar words (should have high similarity):")
    for w1, w2 in similar_pairs:
        sim = embedding_demo.cosine_similarity(w1, w2)
        print(f"  {w1:8} ↔ {w2:8}: {sim:+.3f}")
    print()
    
    # Different pairs
    different_pairs = [
        ('cat', 'mat'),
        ('dog', 'sleep'),
        ('soft', 'table'),
        ('warm', 'cold'),
    ]
    
    print("Different words (should have lower similarity):")
    for w1, w2 in different_pairs:
        sim = embedding_demo.cosine_similarity(w1, w2)
        print(f"  {w1:8} ↔ {w2:8}: {sim:+.3f}")
    print()


def demonstrate_analogies(embedding_demo: SimpleEmbeddingDemo) -> None:
    """Demonstrate word analogies."""
    print("=" * 70)
    print("🔗 WORD ANALOGIES (King - Man + Woman ≈ Queen)")
    print("=" * 70)
    print()
    
    print("Vector math in embedding space:")
    print("  If cat : kitten :: dog : ?")
    print("  Then target = dog + (kitten - cat)")
    print()
    
    analogies = [
        ('cat', 'kitten', 'dog'),
        ('mat', 'soft', 'bed'),
        ('dog', 'run', 'cat'),
    ]
    
    for a, b, c in analogies:
        result = embedding_demo.analogy(a, b, c)
        print(f"  {a:6} is to {b:6} as {c:6} is to: {result}")
        print(f"    → Vector: {c} + ({b} - {a}) ≈ {result}")
        print()


def explain_embedding_concept() -> None:
    """Explain the embedding concept."""
    print("=" * 70)
    print("💡 WHAT ARE EMBEDDINGS?")
    print("=" * 70)
    print()
    print("Embeddings = mapping discrete tokens to continuous vectors")
    print()
    print("Key Properties:")
    print("  1. SEMANTIC SIMILARITY")
    print("     Similar words → similar vectors")
    print("     Example: 'cat' and 'dog' vectors point in similar directions")
    print()
    print("  2. RELATIONSHIPS ARE DIRECTIONS")
    print("     Vector differences capture semantic relationships")
    print("     Example: kitten - cat ≈ puppy - dog (young vs adult)")
    print()
    print("  3. HIGH-DIMENSIONAL SPACE")
    print("     Real LLMs use 512, 768, or 4096+ dimensions")
    print("     We visualize in 2D using PCA/t-SNE")
    print()
    print("In your layers.py (line 76-78):")
    print("  layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)")
    print("  → Creates a learnable matrix of size [vocab_size × embedding_dim]")
    print("  → Each row is a word's vector representation")
    print()


def main():
    """Main execution."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "WEEK 2: EMBEDDING VISUALIZATION" + " " * 21 + "║")
    print("║" + " " * 15 + "Seeing Words as Points in Space" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Vocabulary
    vocab = [
        'cat', 'dog', 'pet', 'animal', 'kitten', 'puppy',
        'mat', 'bed', 'chair', 'table', 'furniture', 'sofa',
        'sit', 'sleep', 'run', 'play', 'walk', 'eat',
        'soft', 'warm', 'hard', 'rough', 'smooth', 'cold',
    ]
    
    # Create embedding demo
    embedding_demo = SimpleEmbeddingDemo(vocab, embedding_dim=16)
    
    # Run demonstrations
    explain_embedding_concept()
    visualize_embedding_space(embedding_demo)
    demonstrate_similarity(embedding_demo)
    demonstrate_analogies(embedding_demo)
    
    print("=" * 70)
    print("🎯 WEEK 2 EMBEDDING TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. EMBEDDINGS = CONTINUOUS MEANING")
    print("   • Discrete words → continuous vectors")
    print("   • Enables neural networks to process text")
    print()
    print("2. SEMANTIC STRUCTURE EMERGES")
    print("   • Similar words cluster together in vector space")
    print("   • Relationships become vector arithmetic")
    print()
    print("3. HIGH DIMENSIONALITY")
    print("   • Our demo: 16 dimensions")
    print("   • Real LLMs: 768 (BERT-base), 4096 (GPT-3), 8192 (GPT-4)")
    print()
    print("4. IN YOUR CODEBASE:")
    print("   • layers.py line 76: layers.Embedding(...)")
    print("   • Creates learnable lookup table [vocab_size × dim]")
    print("   • Trained via backpropagation during model training")
    print()
    print("=" * 70)
    print()
    print("Next: Compare with real embeddings from pre-trained models!")
    print("      Or explore your embeddings/ directory.")
    print()


if __name__ == "__main__":
    main()
