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

"""Week 7: Advanced Architecture - MQA, GQA, and RoPE.

This script demonstrates modern attention mechanisms that improve efficiency:
1. Multi-Query Attention (MQA) - Share K, V across heads
2. Grouped Query Attention (GQA) - Middle ground
3. RoPE (Rotary Position Embeddings) - Modern position encoding

These are used in Llama, Mistral, Gemma, and other 2024-2025 models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import sys

sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')


class AttentionMechanismComparison:
    """Compares MHA, MQA, and GQA attention mechanisms."""
    
    def __init__(self, batch_size: int = 2, seq_len: int = 16,
                 num_heads: int = 8, head_dim: int = 64):
        """Initialize with standard dimensions."""
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
    
    def calculate_memory_bandwidth(self) -> dict:
        """Calculate memory and bandwidth for each attention type."""
        print("=" * 70)
        print("🧠 ATTENTION MECHANISM COMPARISON")
        print("=" * 70)
        print()
        
        # Multi-Head Attention (MHA) - Standard
        # Each head has separate Q, K, V projections
        mha_qkv_params = 3 * self.embed_dim * self.embed_dim  # Q, K, V matrices
        mha_output_params = self.embed_dim * self.embed_dim
        mha_total_params = mha_qkv_params + mha_output_params
        
        # KV cache size per token
        mha_kv_cache_per_token = self.num_heads * self.head_dim * 2  # K + V
        
        # Multi-Query Attention (MQA)
        # All heads share same K, V
        mqa_q_params = self.embed_dim * self.embed_dim  # Q for all heads
        mqa_kv_params = 2 * self.embed_dim * self.head_dim  # Shared K, V (1 head)
        mqa_output_params = self.embed_dim * self.embed_dim
        mqa_total_params = mqa_q_params + mqa_kv_params + mqa_output_params
        
        # KV cache size per token
        mqa_kv_cache_per_token = 1 * self.head_dim * 2  # Single K, V
        
        # Grouped Query Attention (GQA)
        # Groups of heads share K, V
        num_kv_heads = self.num_heads // 2  # 4 groups for 8 heads
        gqa_q_params = self.embed_dim * self.embed_dim
        gqa_kv_params = 2 * (num_kv_heads * self.head_dim) * self.head_dim
        gqa_output_params = self.embed_dim * self.embed_dim
        gqa_total_params = gqa_q_params + gqa_kv_params + gqa_output_params
        
        # KV cache size per token
        gqa_kv_cache_per_token = num_kv_heads * self.head_dim * 2
        
        # Print comparison
        print("MEMORY AND PARAMETERS COMPARISON")
        print()
        print(f"Configuration: {self.num_heads} heads × {self.head_dim} dim = {self.embed_dim} total dim")
        print()
        
        print("┌────────────────────┬─────────────┬─────────────┬─────────────┐")
        print("│ Metric             │ MHA         │ GQA (4 grps)│ MQA         │")
        print("├────────────────────┼─────────────┼─────────────┼─────────────┤")
        print(f"│ Total Parameters   │ {mha_total_params:>10,} │ {gqa_total_params:>10,} │ {mqa_total_params:>10,} │")
        print(f"│ KV Cache / Token   │ {mha_kv_cache_per_token:>10,} │ {gqa_kv_cache_per_token:>10,} │ {mqa_kv_cache_per_token:>10,} │")
        print("└────────────────────┴─────────────┴─────────────┴─────────────┘")
        print()
        
        # Calculate savings
        gqa_param_savings = (1 - gqa_total_params / mha_total_params) * 100
        mqa_param_savings = (1 - mqa_total_params / mha_total_params) * 100
        gqa_cache_savings = (1 - gqa_kv_cache_per_token / mha_kv_cache_per_token) * 100
        mqa_cache_savings = (1 - mqa_kv_cache_per_token / mha_kv_cache_per_token) * 100
        
        print("SAVINGS:")
        print(f"  GQA: {gqa_param_savings:.1f}% fewer params, {gqa_cache_savings:.1f}% less KV cache")
        print(f"  MQA: {mqa_param_savings:.1f}% fewer params, {mqa_cache_savings:.1f}% less KV cache")
        print()
        
        return {
            'mha': {'params': mha_total_params, 'kv_cache': mha_kv_cache_per_token},
            'gqa': {'params': gqa_total_params, 'kv_cache': gqa_kv_cache_per_token,
                   'param_savings': gqa_param_savings, 'cache_savings': gqa_cache_savings},
            'mqa': {'params': mqa_total_params, 'kv_cache': mqa_kv_cache_per_token,
                   'param_savings': mqa_param_savings, 'cache_savings': mqa_cache_savings}
        }
    
    def visualize_attention_mechanisms(self) -> None:
        """Visualize how MHA, GQA, and MQA work."""
        print("=" * 70)
        print("👁️  VISUALIZING ATTENTION MECHANISMS")
        print("=" * 70)
        print()
        
        print("MULTI-HEAD ATTENTION (MHA) - Standard:")
        print("  Head 0: Q0 × K0 → Attention → Output 0")
        print("  Head 1: Q1 × K1 → Attention → Output 1")
        print("  Head 2: Q2 × K2 → Attention → Output 2")
        print("  Head 3: Q3 × K3 → Attention → Output 3")
        print("  ...")
        print("  Concatenate all outputs")
        print("  • Each head has separate K, V projections")
        print("  • High memory, best quality")
        print()
        
        print("GROUPED QUERY ATTENTION (GQA) - Balanced:")
        print("  Group 0: Q0,Q1 × K0 → Attention → Output 0,1")
        print("  Group 1: Q2,Q3 × K1 → Attention → Output 2,3")
        print("  Group 2: Q4,Q5 × K2 → Attention → Output 4,5")
        print("  Group 3: Q6,Q7 × K3 → Attention → Output 6,7")
        print("  • Heads in same group share K, V")
        print("  • Middle ground: memory vs quality")
        print()
        
        print("MULTI-QUERY ATTENTION (MQA) - Fastest:")
        print("  All heads: Q0,Q1,Q2,Q3 × K → Attention → All outputs")
        print("  • All heads share single K, V")
        print("  • Fastest inference, lowest memory")
        print("  • Slight quality degradation")
        print()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        mechanisms = ['MHA', 'GQA', 'MQA']
        kv_cache_sizes = [1024, 512, 128]  # Example values
        param_counts = [524288, 393216, 266240]  # Example values
        
        # KV Cache comparison
        ax1 = axes[0]
        bars1 = ax1.bar(mechanisms, kv_cache_sizes, color=['blue', 'green', 'orange'], alpha=0.7)
        ax1.set_ylabel('KV Cache Size per Token')
        ax1.set_title('KV Cache Memory\n(Lower is Better)')
        ax1.set_ylim(0, 1200)
        for bar, val in zip(bars1, kv_cache_sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{val}', ha='center', fontweight='bold')
        
        # Parameter count comparison
        ax2 = axes[1]
        bars2 = ax2.bar(mechanisms, [p/1000 for p in param_counts], 
                       color=['blue', 'green', 'orange'], alpha=0.7)
        ax2.set_ylabel('Parameters (K)')
        ax2.set_title('Total Parameters\n(Lower is Better)')
        for bar, val in zip(bars2, param_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val/1000:.0f}K', ha='center', fontweight='bold')
        
        # Quality vs Speed trade-off
        ax3 = axes[2]
        quality_scores = [100, 95, 90]  # Relative quality
        speed_scores = [60, 80, 100]  # Relative speed
        
        x = np.arange(len(mechanisms))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, quality_scores, width, label='Quality', 
                        color='purple', alpha=0.7)
        bars3b = ax3.bar(x + width/2, speed_scores, width, label='Speed',
                        color='red', alpha=0.7)
        
        ax3.set_ylabel('Score (Relative %)')
        ax3.set_title('Quality vs Speed Trade-off')
        ax3.set_xticks(x)
        ax3.set_xticklabels(mechanisms)
        ax3.legend()
        ax3.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig('attention_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'attention_comparison.png'")
        print()


class RoPEExplanation:
    """Explains Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, dim: int = 64, max_seq_len: int = 512):
        """Initialize RoPE."""
        self.dim = dim
        self.max_seq_len = max_seq_len
    
    def compute_rotation_angles(self) -> np.ndarray:
        """Compute RoPE rotation angles."""
        # RoPE uses pairs of dimensions
        # For each pair, compute frequency based on position
        
        # Dimension indices (0, 2, 4, ... for first of each pair)
        dim_indices = np.arange(0, self.dim, 2)
        
        # Frequencies: θ_i = 10000^(-2i/dim)
        freqs = 1.0 / (10000 ** (dim_indices / self.dim))
        
        # Position indices
        positions = np.arange(self.max_seq_len)
        
        # Compute angles: angle = position × frequency
        angles = np.outer(positions, freqs)  # [seq_len, dim/2]
        
        return angles
    
    def apply_rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Apply RoPE to input tensor.
        
        Args:
            x: Input tensor [..., dim]
            positions: Position indices [...]
            
        Returns:
            Rotated tensor
        """
        # Split into pairs
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Compute angles for these positions
        angles = self.compute_rotation_angles()
        theta = angles[positions]  # [..., dim/2]
        
        # Apply rotation
        # [x1, x2] rotated by angle θ:
        # x1' = x1 × cos(θ) - x2 × sin(θ)
        # x2' = x1 × sin(θ) + x2 × cos(θ)
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x1_rot = x1 * cos_theta - x2 * sin_theta
        x2_rot = x1 * sin_theta + x2 * cos_theta
        
        # Interleave back
        result = np.empty_like(x)
        result[..., ::2] = x1_rot
        result[..., 1::2] = x2_rot
        
        return result
    
    def visualize_rope(self) -> None:
        """Visualize RoPE mechanism."""
        print("=" * 70)
        print("🔄 ROTARY POSITION EMBEDDINGS (RoPE)")
        print("=" * 70)
        print()
        
        print("THE PROBLEM WITH SINUSOIDAL ENCODING:")
        print("  • Added to embeddings (not part of attention computation)")
        print("  • Position info can get lost in deep layers")
        print("  • Not relative - absolute positions only")
        print()
        
        print("ROPE SOLUTION:")
        print("  • Rotate query/key vectors by position-dependent angle")
        print("  • Position is inherent in the rotation")
        print("  • Maintains relative distances in attention scores")
        print("  • Used in: Llama, Mistral, Gemma, PaLM")
        print()
        
        print("MATHEMATICS:")
        print("  For each pair of dimensions (d, d+1):")
        print("    Frequency: θ = position × (10000^(-2d/dim))")
        print("    Rotation:")
        print("      x_d'   = x_d × cos(θ) - x_{d+1} × sin(θ)")
        print("      x_{d+1}' = x_d × sin(θ) + x_{d+1} × cos(θ)")
        print()
        
        print("VISUALIZATION:")
        print("  Before RoPE:  x = [2.0, 1.0] (at position 0)")
        print("  After RoPE:   x = [1.8, 1.3] (rotated by small angle)")
        print()
        print("  Same vector at position 100:")
        print("  After RoPE:   x = [0.5, 2.1] (rotated by larger angle)")
        print()
        print("  Key insight: Dot product Q·K naturally encodes relative position!")
        print()
        
        # Visualize rotation at different positions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Rotation angles across positions
        ax1 = axes[0, 0]
        angles = self.compute_rotation_angles()[:100]  # First 100 positions
        for i in range(min(4, angles.shape[1])):
            ax1.plot(angles[:, i], label=f'Dim pair {i}', alpha=0.7)
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Rotation Angle (radians)')
        ax1.set_title('RoPE Rotation Angles Across Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 2D rotation visualization
        ax2 = axes[0, 1]
        
        # Original vector
        vec = np.array([1.0, 0.5])
        positions = [0, 10, 50, 100]
        colors = ['blue', 'green', 'orange', 'red']
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        # Plot original
        ax2.arrow(0, 0, vec[0], vec[1], head_width=0.05, color='black', 
                 label='Original', alpha=0.5)
        
        # Plot rotated vectors
        for pos, color in zip(positions, colors):
            # Compute rotation for this position (simplified)
            angle = pos * 0.1  # Simplified angle
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            vec_rot = rot_matrix @ vec
            ax2.arrow(0, 0, vec_rot[0], vec_rot[1], head_width=0.05,
                     color=color, label=f'Position {pos}')
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('2D Rotation by Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Relative position encoding
        ax3 = axes[1, 0]
        
        # Simulate attention scores with RoPE
        seq_len = 20
        query_pos = 10
        
        # Without RoPE: all positions attend equally (no position bias)
        scores_no_rope = np.ones(seq_len)
        
        # With RoPE: nearby positions have higher attention
        distances = np.abs(np.arange(seq_len) - query_pos)
        scores_with_rope = np.exp(-distances / 5)  # Exponential decay with distance
        
        ax3.bar(range(seq_len), scores_no_rope, alpha=0.5, label='No RoPE', color='gray')
        ax3.bar(range(seq_len), scores_with_rope, alpha=0.7, label='With RoPE', color='blue')
        ax3.axvline(x=query_pos, color='red', linestyle='--', label='Query position')
        ax3.set_xlabel('Key Position')
        ax3.set_ylabel('Attention Score')
        ax3.set_title('Relative Position Bias in Attention')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparison with sinusoidal
        ax4 = axes[1, 1]
        
        positions = np.arange(128)
        
        # Sinusoidal encoding (absolute)
        sin_enc = np.sin(positions / 10)
        
        # RoPE effect (relative - shows distance)
        rope_effect = np.cos(positions / 10)  # Simplified
        
        ax4.plot(positions, sin_enc, label='Sinusoidal (absolute)', alpha=0.7)
        ax4.plot(positions, rope_effect, label='RoPE (relative)', alpha=0.7)
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Encoding Value')
        ax4.set_title('RoPE vs Sinusoidal Encoding')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rope_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'rope_visualization.png'")
        print()
    
    def explain_rope_benefits(self) -> None:
        """Explain benefits of RoPE."""
        print("=" * 70)
        print("✨ ROPE BENEFITS")
        print("=" * 70)
        print()
        
        print("1. RELATIVE POSITION ENCODING")
        print("   • Attention score naturally depends on distance between tokens")
        print("   • Dot(Q_rotated, K_rotated) encodes relative position")
        print("   • No learned position embeddings needed!")
        print()
        
        print("2. EXTRAPOLATION TO LONGER SEQUENCES")
        print("   • Can extend to longer sequences than trained on")
        print("   • Llama 2 trained on 4K, works on 32K+ with RoPE scaling")
        print("   • Sinusoidal: struggles with extrapolation")
        print()
        
        print("3. STABILITY IN DEEP LAYERS")
        print("   • Position info preserved through all layers")
        print("   • Unlike additive embeddings that can get 'washed out'")
        print()
        
        print("4. EFFICIENCY")
        print("   • No extra parameters (unlike learned position embeddings)")
        print("   • Simple rotation operation (fast on GPU)")
        print()
        
        print("ADOPTION:")
        print("   • Llama 1/2/3: RoPE")
        print("   • Mistral: RoPE")
        print("   • Gemma: RoPE")
        print("   • PaLM: RoPE")
        print("   • GPT-4: Likely RoPE (not confirmed)")
        print()


class ModernArchitectureSummary:
    """Summarize modern architecture choices."""
    
    def summarize_2024_models(self) -> None:
        """Summarize 2024 model architectures."""
        print("=" * 70)
        print("🏗️  MODERN ARCHITECTURE CHOICES (2024-2025)")
        print("=" * 70)
        print()
        
        print("┌─────────────────┬─────────────┬─────────────┬─────────────┐")
        print("│ Model           │ Attention   │ Pos Encoding│ Norm        │")
        print("├─────────────────┼─────────────┼─────────────┼─────────────┤")
        print("│ Llama 2 7B      │ GQA (4 grps)│ RoPE        │ RMSNorm     │")
        print("│ Llama 2 70B     │ GQA (8 grps)│ RoPE        │ RMSNorm     │")
        print("│ Llama 3 8B      │ GQA (4 grps)│ RoPE        │ RMSNorm     │")
        print("│ Llama 3 70B     │ GQA (8 grps)│ RoPE        │ RMSNorm     │")
        print("│ Mistral 7B      │ GQA (4 grps)│ RoPE        │ RMSNorm     │")
        print("│ Gemma 7B        │ MQA         │ RoPE        │ RMSNorm     │")
        print("│ GPT-3 (2020)    │ MHA         │ Learned     │ LayerNorm   │")
        print("│ GPT-4 (2023)    │ ?           │ ?           │ ?           │")
        print("└─────────────────┴─────────────┴─────────────┴─────────────┘")
        print()
        
        print("TRENDS:")
        print("  • GQA/MQA replacing MHA (memory efficiency)")
        print("  • RoPE replacing sinusoidal/learned (better extrapolation)")
        print("  • RMSNorm replacing LayerNorm (faster, similar quality)")
        print("  • Pre-norm becoming standard (better stability)")
        print()
        
        print("YOUR CODEBASE STATUS:")
        print("  • Attention: MHA (can add GQA/MQA)")
        print("  • Position: Sinusoidal (can upgrade to RoPE)")
        print("  • Norm: LayerNorm (can upgrade to RMSNorm)")
        print("  • Architecture: Post-norm (can switch to Pre-norm)")
        print()
        
        print("UPGRADE PATH:")
        print("  1. Implement GQA in attention/ directory")
        print("  2. Add RoPE to embeddings/")
        print("  3. Try RMSNorm in place of LayerNorm")
        print("  4. Switch to Pre-norm architecture")
        print()


def main():
    """Main execution for Week 7."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "WEEK 7: ADVANCED ARCHITECTURE" + " " * 19 + "║")
    print("║" + " " * 15 + "MQA, GQA, and RoPE (Modern Standards)" + " " * 14 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Attention mechanisms
    attention = AttentionMechanismComparison()
    attention.calculate_memory_bandwidth()
    attention.visualize_attention_mechanisms()
    
    # RoPE
    rope = RoPEExplanation()
    rope.visualize_rope()
    rope.explain_rope_benefits()
    
    # Summary
    summary = ModernArchitectureSummary()
    summary.summarize_2024_models()
    
    print("=" * 70)
    print("🎯 WEEK 7 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. MULTI-QUERY ATTENTION (MQA):")
    print("   • All heads share single K, V")
    print("   • 8× KV cache reduction!")
    print("   • Fastest inference, slight quality trade-off")
    print("   • Used in: Gemma")
    print()
    print("2. GROUPED QUERY ATTENTION (GQA):")
    print("   • Groups of heads share K, V")
    print("   • Middle ground: MHA quality, MQA speed")
    print("   • 2-4× KV cache reduction")
    print("   • Used in: Llama 2/3, Mistral")
    print()
    print("3. ROTARY POSITION EMBEDDINGS (RoPE):")
    print("   • Rotate Q/K by position-dependent angle")
    print("   • Relative position encoding")
    print("   • Excellent extrapolation to longer sequences")
    print("   • Used in: Llama, Mistral, Gemma, PaLM")
    print()
    print("4. MODERN STACK (2024-2025):")
    print("   • Attention: GQA or MQA")
    print("   • Position: RoPE")
    print("   • Normalization: RMSNorm")
    print("   • Architecture: Pre-norm")
    print()
    print("5. YOUR UPGRADE PATH:")
    print("   • Add GQA to your attention implementation")
    print("   • Replace sinusoidal with RoPE")
    print("   • Try RMSNorm instead of LayerNorm")
    print("   • Switch from Post-norm to Pre-norm")
    print()
    print("=" * 70)
    print()
    print("Next: Week 8 - Normalization & Stability")
    print("      (RMSNorm, Pre-norm vs Post-norm)")
    print()


if __name__ == "__main__":
    main()
