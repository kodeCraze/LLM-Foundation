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

"""Week 1: Attention Visualization - Building Transformer Intuition.

This script demonstrates how attention works through interactive visualizations.
It's designed for visual/intuitive learners who want to understand WHY
transformers work before diving into the math.

Run this script to see:
1. How tokens "look at" each other
2. Attention weights as a heatmap
3. The weighted sum intuition
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Week 1 Goal: Visual Intuition
# Key Insight: Attention is just "weighted averaging" where weights are learned


def create_simple_sentence() -> Tuple[List[str], np.ndarray]:
    """Creates a simple 4-word sentence and dummy attention weights.
    
    Sentence: "The cat sat down"
    
    Returns:
        tokens: List of words
        attention_weights: 4x4 matrix showing which words attend to which
    """
    tokens = ["The", "cat", "sat", "down"]
    
    # Dummy attention pattern (what a well-trained model might learn):
    # - "cat" strongly attends to "The" (subject-determiner agreement)
    # - "sat" attends to both "The" and "cat" (verb needs subject)
    # - "down" attends to "sat" (completes the phrase)
    # - Each word always attends to itself strongly
    
    attention_weights = np.array([
        # The   cat   sat  down
        [0.50, 0.30, 0.15, 0.05],  # The attends to...
        [0.40, 0.45, 0.10, 0.05],  # cat attends to...
        [0.25, 0.35, 0.30, 0.10],  # sat attends to...
        [0.05, 0.05, 0.60, 0.30],  # down attends to...
    ])
    
    return tokens, attention_weights


def visualize_attention_heatmap(tokens: List[str], attention: np.ndarray) -> None:
    """Creates a beautiful heatmap showing attention patterns.
    
    Args:
        tokens: List of tokens (words)
        attention: Attention weight matrix [seq_len, seq_len]
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # Labels
    ax.set_xlabel('Key Tokens (Being Looked At)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Tokens (Looking)', fontsize=12, fontweight='bold')
    ax.set_title('Attention Heatmap: How Each Word "Looks At" Other Words\n'
                 '(Brighter = More Attention)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f'{attention[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Heatmap saved to 'attention_heatmap.png'")


def explain_attention_concept() -> None:
    """Prints intuitive explanation of attention mechanism."""
    print("=" * 70)
    print("🧠 ATTENTION EXPLAINED (The Core Idea)")
    print("=" * 70)
    print()
    print("Think of attention like this:")
    print()
    print("  Imagine you're reading the sentence: 'The cat sat down'")
    print()
    print("  When you read the word 'sat', you subconsciously look back at")
    print("  'cat' to understand WHO is sitting. This 'looking back' is attention.")
    print()
    print("  In transformers:")
    print("  • Each word creates a QUERY: 'What am I looking for?'")
    print("  • Each word creates a KEY: 'What do I contain?'")
    print("  • Attention weight = similarity(Query, Key)")
    print("  • If Query matches Key well → high attention weight")
    print()
    print("  The OUTPUT for each word is a weighted sum of all words,")
    print("  where the weights are the attention scores.")
    print()
    print("  Key Insight: ATTENTION IS JUST WEIGHTED AVERAGING")
    print("  • Traditional averaging: every word counts equally")
    print("  • Attention: important words count more")
    print()
    print("=" * 70)


def demonstrate_weighted_sum() -> None:
    """Shows how attention creates context-aware representations."""
    print("\n" + "=" * 70)
    print("🔢 WEIGHTED SUM DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Simple word embeddings (in reality these are 512+ dimensional)
    embeddings = {
        "The": np.array([1.0, 0.1]),
        "cat": np.array([0.2, 0.9]),
        "sat": np.array([0.8, 0.3]),
        "down": np.array([0.1, 0.8]),
    }
    
    tokens, attention = create_simple_sentence()
    
    print("Word Embeddings (simplified 2D for visualization):")
    for token, emb in embeddings.items():
        print(f"  {token:5} → [{emb[0]:.1f}, {emb[1]:.1f}]")
    print()
    
    # Calculate context vector for "sat" (index 2)
    target_idx = 2
    target_token = tokens[target_idx]
    
    print(f"Computing output for '{target_token}':")
    print(f"  Attention weights: {attention[target_idx]}")
    print()
    
    # Weighted sum
    context_vector = np.zeros(2)
    for i, token in enumerate(tokens):
        weight = attention[target_idx, i]
        contribution = weight * embeddings[token]
        context_vector += contribution
        print(f"  {token} contributes {weight:.2f} × [{embeddings[token][0]:.1f}, {embeddings[token][1]:.1f}] = "
              f"[{contribution[0]:.2f}, {contribution[1]:.2f}]")
    
    print()
    print(f"  Final context vector for '{target_token}': [{context_vector[0]:.2f}, {context_vector[1]:.2f}]")
    print()
    print("  Notice: The context vector blends information from words that")
    print("  'sat' attends to (mainly 'cat' and itself). This is how transformers")
    print("  create context-aware meaning!")
    print()
    print("=" * 70)


def show_causal_masking() -> None:
    """Demonstrates why LLMs use causal (look-ahead) masking."""
    print("\n" + "=" * 70)
    print("🎭 CAUSAL MASKING (Why LLMs Can't Look Ahead)")
    print("=" * 70)
    print()
    print("When generating text, LLMs predict ONE token at a time.")
    print("They should ONLY look at previous tokens, not future ones.")
    print()
    
    tokens = ["The", "cat", "sat", "down"]
    
    # Causal mask (upper triangle is masked/zeroed out)
    causal_mask = np.tril(np.ones((4, 4)))
    
    print("Causal Mask (1 = can look, 0 = cannot look):")
    print("  " + " ".join([f"{t:>4}" for t in tokens]))
    for i, token in enumerate(tokens):
        row = "  ".join([f"{causal_mask[i, j]:4.0f}" for j in range(len(tokens))])
        print(f"{token:4} [{row}]")
    
    print()
    print("Meaning:")
    for i, token in enumerate(tokens):
        can_see = [tokens[j] for j in range(i + 1)]
        print(f"  '{token}' can only look at: {can_see}")
    
    print()
    print("This is why your `MultiHeadSelfAttention` uses `use_causal_mask=True`!")
    print("  → See: layers.py line 341")
    print()
    print("=" * 70)


def connect_to_your_code() -> None:
    """Connects concepts to the actual code in the repository."""
    print("\n" + "=" * 70)
    print("🔗 CONNECTING TO YOUR CODEBASE")
    print("=" * 70)
    print()
    print("Your repository already implements all these concepts!")
    print()
    print("1. EMBEDDINGS (transformers/layers.py:38-156)")
    print("   TokenAndPositionEmbedding combines:")
    print("   • Token embeddings: each word → vector")
    print("   • Positional encoding: WHERE the word is in the sentence")
    print()
    print("2. ATTENTION (transformers/layers.py:298-347)")
    print("   MultiHeadSelfAttention:")
    print("   • Uses Keras MultiHeadAttention (handles the Q/K/V math)")
    print("   • use_causal_mask=True enforces the masking we showed above")
    print("   • LayerNorm + Residual connection for stable training")
    print()
    print("3. FEED-FORWARD (transformers/layers.py:237-293)")
    print("   FeedForwardNetwork:")
    print("   • Two dense layers (expand then project back)")
    print("   • Adds non-linearity (ReLU activation)")
    print("   • LayerNorm + Residual again")
    print()
    print("4. COMPLETE MODEL (training/model.py:32-121)")
    print("   create_model() stacks everything:")
    print("   Input → Embedding → [TransformerBlock × N] → Dense → Output")
    print()
    print("=" * 70)


def main() -> None:
    """Main execution for Week 1 visualization."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "WEEK 1: ATTENTION VISUALIZATION" + " " * 17 + "║")
    print("║" + " " * 15 + "Building Intuition Through Visual Learning" + " " * 13 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Core concept explanation
    explain_attention_concept()
    
    # Create and visualize attention
    tokens, attention = create_simple_sentence()
    visualize_attention_heatmap(tokens, attention)
    
    # Show weighted sum calculation
    demonstrate_weighted_sum()
    
    # Explain causal masking
    show_causal_masking()
    
    # Connect to actual code
    connect_to_your_code()
    
    print("\n" + "=" * 70)
    print("🎯 WEEK 1 CHECKPOINT")
    print("=" * 70)
    print()
    print("Can you now answer these questions?")
    print()
    print("1. What is attention in plain English?")
    print("   → Hint: weighted averaging with learned importance")
    print()
    print("2. Why do we need causal masking in LLMs?")
    print("   → Hint: can't look at future when predicting next token")
    print()
    print("3. What three things does TokenAndPositionEmbedding do?")
    print("   → Hint: token lookup + position info + combination")
    print()
    print("4. Look at layers.py lines 204-209 - what's the transformer block doing?")
    print("   → Hint: attention first, then feed-forward")
    print()
    print("Next: Try modifying the attention weights in this script to see")
    print("      how different patterns affect the output!")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
