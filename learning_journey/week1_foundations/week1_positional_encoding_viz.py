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

"""Week 1: Positional Encoding Visualization.

This script visualizes sinusoidal positional encodings, the exact code
at layers.py:95-130. It answers: "Why do we add position information?"

This is a pure NumPy implementation for educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_sinusoidal_encoding() -> None:
    """Visualizes how sinusoidal positional encoding works."""
    print("=" * 70)
    print("📍 POSITIONAL ENCODING: Why Words Need Position Information")
    print("=" * 70)
    print()
    
    print("THE PROBLEM:")
    print("  Transformers process all tokens AT ONCE (not left-to-right like RNNs)")
    print("  Without position info: 'cat sat the down' = 'the cat sat down'")
    print("  The model can't tell word order!")
    print()
    
    print("THE SOLUTION (from 'Attention Is All You Need' paper):")
    print("  Add a UNIQUE position signature to each position")
    print("  Use sine/cosine waves at different frequencies")
    print()
    
    # Parameters matching layers.py
    max_length = 32
    embedding_dim = 64
    angle_rate_multiplier = 10000  # Same as layers.py:33
    
    print("Configuration (from your layers.py):")
    print(f"  • Max length: {max_length}")
    print(f"  • Embedding dim: {embedding_dim}")
    print(f"  • Angle rate multiplier: {angle_rate_multiplier}")
    print()
    
    # Reproduce the encoding calculation from layers.py:95-130
    depth = embedding_dim // 2
    positions = np.arange(max_length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    
    angle_rates = 1 / (angle_rate_multiplier ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    
    # The encoding
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
    )
    
    print("Mathematical intuition:")
    print("  Position 0:  [sin(0), cos(0), sin(0), cos(0), ...]")
    print("  Position 1:  [sin(1×rate₁), cos(1×rate₁), sin(1×rate₂), ...]")
    print("  Position 2:  [sin(2×rate₁), cos(2×rate₁), sin(2×rate₂), ...]")
    print()
    print("  Key insight: Each position gets a UNIQUE 'fingerprint'")
    print("  The model learns to use these to understand word order!")
    print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Full encoding heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(pos_encoding, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Position in Sequence')
    ax1.set_title('Full Positional Encoding Matrix\n(Red=positive, Blue=negative)')
    ax1.set_yticks([0, 7, 15, 23, 31])
    ax1.set_yticklabels(['0', '7', '15', '23', '31'])
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Specific dimensions across positions
    ax2 = axes[0, 1]
    dimensions_to_plot = [0, 4, 8, 16, 31]
    for dim in dimensions_to_plot:
        if dim < embedding_dim:
            ax2.plot(pos_encoding[:16, dim], label=f'Dim {dim}', alpha=0.7)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Encoding Values vs Position\n(Different dimensions)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, 15)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sine wave visualization for one dimension
    ax3 = axes[1, 0]
    positions_fine = np.linspace(0, max_length, 500)
    rate = angle_rates[0, 8]  # One specific rate
    ax3.plot(positions_fine, np.sin(positions_fine * rate), 'b-', label=f'Sine (dim 8)')
    ax3.plot(positions_fine, np.cos(positions_fine * rate), 'r--', label=f'Cosine (dim 8+32)')
    # Mark actual discrete positions
    ax3.scatter(np.arange(max_length), pos_encoding[:, 8], c='blue', s=30, zorder=5)
    ax3.scatter(np.arange(max_length), pos_encoding[:, 8 + 32], c='red', s=30, zorder=5)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Value')
    ax3.set_title('Sine/Cosine Waves for One Dimension\n(Dots = actual encoding values)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Why sine/cosine? Show linear relationship
    ax4 = axes[1, 1]
    # Show that position encodings have a linear relationship
    # which helps the model learn relative positions
    pos_a = 5
    pos_b = 10
    encoding_a = pos_encoding[pos_a]
    encoding_b = pos_encoding[pos_b]
    
    ax4.plot(encoding_a, label=f'Position {pos_a}', alpha=0.7, marker='o', markersize=2)
    ax4.plot(encoding_b, label=f'Position {pos_b}', alpha=0.7, marker='s', markersize=2)
    ax4.set_xlabel('Embedding Dimension')
    ax4.set_ylabel('Encoding Value')
    ax4.set_title(f'Encoding at Positions {pos_a} and {pos_b}\n(Similar but shifted pattern)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to 'positional_encoding.png'")
    print()


def compare_embedding_types() -> None:
    """Compares learned vs sinusoidal positional embeddings."""
    print("=" * 70)
    print("🔬 COMPARISON: Learned vs Sinusoidal Embeddings")
    print("=" * 70)
    print()
    
    max_length = 16
    embedding_dim = 32
    
    print("Your layers.py supports BOTH embedding types (line 80-93):")
    print()
    
    print("1. SIMPLE (Learned) Embeddings:")
    print("   → Each position has a learnable vector (like word embeddings)")
    print("   → Pros: Can adapt to specific task")
    print("   → Cons: Can't generalize to sequences longer than training")
    print(f"   → Parameters: {max_length * embedding_dim:,} learnable position params")
    print()
    
    print("2. SINUSOIDAL (Fixed) Embeddings:")
    print("   → Deterministic sine/cosine waves (what we visualized above)")
    print("   → Pros: Generalizes to ANY length, no extra parameters!")
    print("   → Cons: Less flexible, but works great in practice")
    print("   → Parameters: 0 (fixed mathematical formula)")
    print()
    
    print("Key difference:")
    print(f"   Simple:     {max_length * embedding_dim:,} position params to learn")
    print(f"   Sinusoidal: 0 position params (just compute sine/cosine)")
    print()
    print("→ Most modern models (GPT, Llama, Gemma) use some form of")
    print("  learned or rotary position embeddings now, but sinusoidal")
    print("  is the classic choice from the original Transformer paper!")
    print()
    print("=" * 70)


def practical_demonstration() -> None:
    """Shows why position matters with a concrete example."""
    print("=" * 70)
    print("💡 PRACTICAL DEMONSTRATION")
    print("=" * 70)
    print()
    
    print("Consider these two sentences:")
    print("  A: 'The cat sat on the mat'")
    print("  B: 'The mat sat on the cat'")
    print()
    print("Same words, different order = COMPLETELY different meaning!")
    print()
    print("Without position encoding:")
    print("  → Transformer sees: {The, cat, sat, on, the, mat}")
    print("  → No way to distinguish A from B")
    print()
    print("With position encoding:")
    print("  → Position 0: The + pos_enc[0]")
    print("  → Position 1: cat + pos_enc[1]")
    print("  → Position 2: sat + pos_enc[2]")
    print("  → ...")
    print()
    print("  For sentence A: embeddings = [The+pe0, cat+pe1, sat+pe2, ...]")
    print("  For sentence B: embeddings = [The+pe0, mat+pe1, sat+pe2, ...]")
    print()
    print("  Now 'mat' at position 1 is DIFFERENT from 'cat' at position 1")
    print("  → Model can tell word order!")
    print()
    print("=" * 70)


def connect_to_code() -> None:
    """Shows exactly where this is in the codebase."""
    print()
    print("=" * 70)
    print("🔗 CODE CONNECTION: Your layers.py")
    print("=" * 70)
    print()
    print("Look at transformers/layers.py:")
    print()
    print("Line 33: ANGLE_RATE_MULTIPLIER = 10000")
    print("  → The 'base' for the exponential decay of frequencies")
    print()
    print("Line 85-93: __init__() chooses between 'simple' and 'sinusoidal'")
    print("  → You can switch between learned and fixed encodings!")
    print()
    print("Line 95-130: positional_encoding() method")
    print("  → This is EXACTLY what we visualized above")
    print("  → Creates the sine/cosine matrix")
    print()
    print("Line 144-149: Scaling in call()")
    print("  → Token embeddings are scaled by sqrt(embedding_dim)")
    print("  → Balances the scale of token vs position embeddings")
    print()
    print("Line 156: return token_embeddings + position_embeddings")
    print("  → The final combination!")
    print()
    print("=" * 70)


def main() -> None:
    """Main execution."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WEEK 1: POSITIONAL ENCODING" + " " * 26 + "║")
    print("║" + " " * 12 + "Understanding Position in Transformers" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Run demonstrations
    visualize_sinusoidal_encoding()
    compare_embedding_types()
    practical_demonstration()
    connect_to_code()
    
    print()
    print("=" * 70)
    print("🎯 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. Position encoding = telling the model WHERE each word is")
    print()
    print("2. Sinusoidal encoding uses sine/cosine waves because:")
    print("   • Each position gets unique 'fingerprint'")
    print("   • Generalizes to longer sequences")
    print("   • No extra parameters to learn")
    print()
    print("3. Your code implements BOTH options:")
    print("   • 'simple' = learned embeddings (like word embeddings)")
    print("   • 'sinusoidal' = fixed math (from original paper)")
    print()
    print("4. The encoding is ADDED to token embeddings (line 156)")
    print("   → Result has both meaning (token) and position info")
    print()
    print("=" * 70)
    print()
    print("Next: Open layers.py and trace through lines 95-156")
    print("      with the visualization fresh in your mind!")
    print()


if __name__ == "__main__":
    main()
