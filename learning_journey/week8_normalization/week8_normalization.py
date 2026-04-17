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

"""Week 8: Normalization & Architecture Stability.

This script demonstrates:
1. LayerNorm vs RMSNorm - differences and trade-offs
2. Pre-norm vs Post-norm architecture
3. Why modern models use Pre-norm + RMSNorm
4. Training stability techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys

sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')


class NormalizationComparison:
    """Compares different normalization techniques."""
    
    def __init__(self):
        """Initialize."""
        pass
    
    def layernorm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply LayerNorm.
        
        LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + eps)
        
        # In practice, gamma and beta are learned parameters
        # For visualization, we use identity (gamma=1, beta=0)
        return x_norm
    
    def rmsnorm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply RMSNorm.
        
        RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
        
        Note: No mean subtraction, no beta parameter
        """
        # Compute root mean square
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        
        # Normalize
        x_norm = x / rms
        
        return x_norm
    
    def compare_normalizations(self) -> None:
        """Compare LayerNorm and RMSNorm."""
        print("=" * 70)
        print("📊 LAYER NORM vs RMS NORM")
        print("=" * 70)
        print()
        
        print("LAYER NORM (LayerNorm):")
        print("  Formula: (x - mean) / sqrt(var + eps) * gamma + beta")
        print("  Steps:")
        print("    1. Compute mean across features")
        print("    2. Compute variance")
        print("    3. Subtract mean, divide by std")
        print("    4. Scale by gamma, shift by beta")
        print()
        print("  Parameters: 2 * dim (gamma + beta)")
        print("  Used in: Original Transformer, BERT, GPT-2/3")
        print()
        
        print("RMS NORM (RMSNorm):")
        print("  Formula: x / sqrt(mean(x^2) + eps) * gamma")
        print("  Steps:")
        print("    1. Compute root mean square (no mean subtraction!)")
        print("    2. Divide by RMS")
        print("    3. Scale by gamma (no beta!)")
        print()
        print("  Parameters: dim (gamma only)")
        print("  Used in: Llama, Mistral, Gemma (all 2024 models)")
        print()
        
        print("KEY DIFFERENCES:")
        print("  LayerNorm:")
        print("    ❌ Subtracts mean (extra computation)")
        print("    ❌ Has beta parameter (learned shift)")
        print("    ✅ More flexible (centers and scales)")
        print()
        print("  RMSNorm:")
        print("    ✅ No mean subtraction (faster)")
        print("    ✅ No beta (fewer parameters)")
        print("    ✅ Similar performance in practice")
        print()
        
        # Visualize effect on data
        np.random.seed(42)
        data = np.random.randn(1000, 64) * 5 + 10  # Mean=10, Std=5
        
        layernorm_data = self.layernorm(data)
        rmsnorm_data = self.rmsnorm(data)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original data distribution
        ax = axes[0, 0]
        ax.hist(data.flatten(), bins=50, alpha=0.7, color='gray')
        ax.set_title('Original Data\nMean=10, Std=5')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # LayerNorm result
        ax = axes[0, 1]
        ax.hist(layernorm_data.flatten(), bins=50, alpha=0.7, color='blue')
        ax.set_title('After LayerNorm\nMean≈0, Std≈1')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # RMSNorm result
        ax = axes[0, 2]
        ax.hist(rmsnorm_data.flatten(), bins=50, alpha=0.7, color='green')
        ax.set_title('After RMSNorm\nMean≠0, RMS≈1')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Comparison of means
        ax = axes[1, 0]
        sample_indices = np.arange(100)
        ax.plot(sample_indices, data[:100].mean(axis=1), 
               label='Original', alpha=0.7, color='gray')
        ax.plot(sample_indices, layernorm_data[:100].mean(axis=1),
               label='LayerNorm', alpha=0.7, color='blue')
        ax.plot(sample_indices, rmsnorm_data[:100].mean(axis=1),
               label='RMSNorm', alpha=0.7, color='green')
        ax.set_title('Mean Across Samples')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Mean')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Comparison of std/RMS
        ax = axes[1, 1]
        original_std = data[:100].std(axis=1)
        layernorm_std = layernorm_data[:100].std(axis=1)
        rmsnorm_rms = np.sqrt(np.mean(rmsnorm_data[:100]**2, axis=1))
        
        ax.plot(sample_indices, original_std, label='Original Std', alpha=0.7, color='gray')
        ax.plot(sample_indices, layernorm_std, label='LayerNorm Std', alpha=0.7, color='blue')
        ax.plot(sample_indices, rmsnorm_rms, label='RMSNorm RMS', alpha=0.7, color='green')
        ax.set_title('Standard Deviation / RMS')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Std / RMS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Computational comparison
        ax = axes[1, 2]
        
        operations = ['Mean', 'Var', 'Sub', 'Div', 'Scale', 'Shift']
        layernorm_ops = [1, 1, 1, 1, 1, 1]
        rmsnorm_ops = [0, 1, 0, 1, 1, 0]
        
        x = np.arange(len(operations))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, layernorm_ops, width, label='LayerNorm', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, rmsnorm_ops, width, label='RMSNorm', color='green', alpha=0.7)
        
        ax.set_ylabel('Required')
        ax.set_title('Operations Required')
        ax.set_xticks(x)
        ax.set_xticklabels(operations)
        ax.legend()
        ax.set_ylim(0, 1.5)
        
        plt.tight_layout()
        plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'normalization_comparison.png'")
        print()
        
        print("COMPUTATIONAL COMPLEXITY:")
        print("  LayerNorm: ~2N operations (mean + var + normalize)")
        print("  RMSNorm:   ~N operations (rms + normalize)")
        print("  Speedup:   ~30-40% faster (in practice)")
        print()


class ArchitectureComparison:
    """Compares Pre-norm vs Post-norm architectures."""
    
    def __init__(self):
        """Initialize."""
        pass
    
    def explain_architectures(self) -> None:
        """Explain Pre-norm and Post-norm."""
        print("=" * 70)
        print("🏗️  PRE-NORM vs POST-NORM ARCHITECTURE")
        print("=" * 70)
        print()
        
        print("POST-NORM (Original Transformer):")
        print("  x = x + Attention(Norm(x))")
        print("  x = x + FFN(Norm(x))")
        print()
        print("  Order: Norm → Sub-layer → Residual")
        print()
        print("  Visualization:")
        print("    Input")
        print("      ↓")
        print("    [Norm]")
        print("      ↓")
        print("    [Attention]")
        print("      ↓")
        print("    + Residual")
        print("      ↓")
        print("    Output")
        print()
        print("  Problem: Gradients can explode in deep networks!")
        print("  Used in: Original Transformer, BERT, GPT-2")
        print()
        
        print("PRE-NORM (Modern Standard):")
        print("  x = x + Attention(x)")
        print("  x = Norm(x)")
        print("  x = x + FFN(x)")
        print("  x = Norm(x)")
        print()
        print("  Order: Sub-layer → Residual → Norm")
        print()
        print("  Visualization:")
        print("    Input")
        print("      ↓")
        print("    [Attention]")
        print("      ↓")
        print("    + Residual")
        print("      ↓")
        print("    [Norm]")
        print("      ↓")
        print("    Output")
        print()
        print("  Benefit: More stable gradients, easier to train deep!")
        print("  Used in: Llama, Mistral, GPT-3, all 2024 models")
        print()
        
        print("WHY PRE-NORM IS BETTER:")
        print("  1. Gradient flow:")
        print("     • Post-norm: Gradient goes through Norm layer")
        print("     • Pre-norm: Gradient has direct path (residual)")
        print()
        print("  2. Warmup requirements:")
        print("     • Post-norm: Needs careful warmup")
        print("     • Pre-norm: More forgiving")
        print()
        print("  3. Deep networks:")
        print("     • Post-norm: Struggles beyond ~12 layers")
        print("     • Pre-norm: Works with 100+ layers!")
        print()
    
    def visualize_gradient_flow(self) -> None:
        """Visualize gradient flow in both architectures."""
        print("=" * 70)
        print("📉 GRADIENT FLOW VISUALIZATION")
        print("=" * 70)
        print()
        
        # Simulate gradient magnitudes through layers
        np.random.seed(42)
        num_layers = 24
        
        # Post-norm: gradients can explode/vanish
        post_norm_grads = []
        grad = 1.0
        for i in range(num_layers):
            # Gradient goes through normalization (unstable)
            grad = grad * (1 + np.random.randn() * 0.3)  # Multiplicative noise
            grad = np.clip(grad, 0.1, 5.0)  # Prevent total explosion
            post_norm_grads.append(grad)
        
        # Pre-norm: more stable gradient flow
        pre_norm_grads = []
        grad = 1.0
        for i in range(num_layers):
            # Direct residual connection helps
            grad = 0.9 * grad + 0.1 * np.random.randn()  # Mostly preserved
            grad = np.clip(grad, 0.5, 2.0)
            pre_norm_grads.append(grad)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Post-norm
        layers = np.arange(num_layers)
        ax1.plot(layers, post_norm_grads, 'b-o', alpha=0.7, linewidth=2)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Gradient Magnitude')
        ax1.set_title('Post-Norm: Unstable Gradient Flow')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 6)
        
        # Pre-norm
        ax2.plot(layers, pre_norm_grads, 'g-o', alpha=0.7, linewidth=2)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Gradient Magnitude')
        ax2.set_title('Pre-Norm: Stable Gradient Flow')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 6)
        
        plt.tight_layout()
        plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'gradient_flow.png'")
        print()
        
        print("OBSERVATION:")
        print("  Post-norm: Gradient varies wildly (unstable training)")
        print("  Pre-norm:  Gradient stays near 1.0 (stable training)")
        print()
    
    def compare_architecture_choices(self) -> None:
        """Compare architecture choices of modern models."""
        print("=" * 70)
        print("📊 MODERN ARCHITECTURE CHOICES")
        print("=" * 70)
        print()
        
        print("┌─────────────────┬─────────────┬─────────────┬─────────────┐")
        print("│ Model           │ Norm Type   │ Norm Place  │ Activation  │")
        print("├─────────────────┼─────────────┼─────────────┼─────────────┤")
        print("│ GPT-2 (2019)    │ LayerNorm   │ Post-norm   │ GELU        │")
        print("│ GPT-3 (2020)    │ LayerNorm   │ Pre-norm    │ GELU        │")
        print("│ BERT (2019)     │ LayerNorm   │ Post-norm   │ GELU        │")
        print("│ T5 (2020)       │ LayerNorm   │ Pre-norm    │ ReLU        │")
        print("│ Llama 1 (2023)  │ RMSNorm     │ Pre-norm    │ SwiGLU      │")
        print("│ Llama 2 (2023)  │ RMSNorm     │ Pre-norm    │ SwiGLU      │")
        print("│ Llama 3 (2024)  │ RMSNorm     │ Pre-norm    │ SwiGLU      │")
        print("│ Mistral (2023)  │ RMSNorm     │ Pre-norm    │ SwiGLU      │")
        print("│ Gemma (2024)    │ RMSNorm     │ Pre-norm    │ GELU        │")
        print("└─────────────────┴─────────────┴─────────────┴─────────────┘")
        print()
        
        print("TRENDS (2019 → 2024):")
        print("  • LayerNorm → RMSNorm (faster, similar quality)")
        print("  • Post-norm → Pre-norm (more stable)")
        print("  • GELU/ReLU → SwiGLU (better performance)")
        print()
        
        print("THE MODERN STACK (2024):")
        print("  ✅ RMSNorm (faster than LayerNorm)")
        print("  ✅ Pre-norm (more stable)")
        print("  ✅ SwiGLU activation (better than GELU)")
        print("  ✅ GQA attention (memory efficient)")
        print("  ✅ RoPE position (extrapolates well)")
        print()


class TrainingStabilityTechniques:
    """Demonstrates training stability techniques."""
    
    def explain_stability_techniques(self) -> None:
        """Explain techniques for training stability."""
        print("=" * 70)
        print("🛡️  TRAINING STABILITY TECHNIQUES")
        print("=" * 70)
        print()
        
        print("1. RESIDUAL CONNECTIONS (Skip Connections)")
        print("   Formula: output = layer(x) + x")
        print()
        print("   Why it helps:")
        print("     • Gradient can flow directly through the skip")
        print("     • Even if layer gradients vanish, skip preserves signal")
        print("     • Enables training of very deep networks (100+ layers)")
        print()
        print("   Visualization:")
        print("     Input ──┬──→ [Layer] ──┐")
        print("             │               │")
        print("             └──→ (skip) ───┘")
        print("                          ↓")
        print("                       Output")
        print()
        
        print("2. NORMALIZATION PLACEMENT (Pre-norm)")
        print("   Pre-norm: x = x + Sublayer(Norm(x))")
        print()
        print("   Why it helps:")
        print("     • Keeps activations in reasonable range")
        print("     • Prevents exploding/vanishing gradients")
        print("     • More forgiving hyperparameters")
        print()
        
        print("3. GRADIENT CLIPPING")
        print("   If gradient_norm > threshold: scale down")
        print()
        print("   Why it helps:")
        print("     • Prevents gradient explosion")
        print("     • Common threshold: 1.0")
        print("     • Essential for transformers")
        print()
        
        print("4. LEARNING RATE WARMUP")
        print("   Start with small LR, gradually increase")
        print()
        print("   Why it helps:")
        print("     • Prevents early training instability")
        print("     • Lets model settle before full optimization")
        print("     • Typical: 1-2% of total steps")
        print()
        
        print("5. WEIGHT INITIALIZATION")
        print("   Careful initialization prevents early instability")
        print()
        print("   Common strategies:")
        print("     • Xavier/Glorot: For layers with symmetric activation")
        print("     • He: For ReLU-like activations")
        print("     • Small std for embeddings (prevent early saturation)")
        print()
    
    def visualize_stability_comparison(self) -> None:
        """Visualize training stability with/without techniques."""
        print("=" * 70)
        print("📈 TRAINING STABILITY COMPARISON")
        print("=" * 70)
        print()
        
        # Simulate training loss curves
        np.random.seed(42)
        epochs = np.arange(100)
        
        # Ideal training (all techniques)
        ideal_loss = 3.0 * np.exp(-epochs / 20) + 0.5
        ideal_loss += np.random.randn(100) * 0.05
        
        # Without Pre-norm (unstable)
        unstable_loss = 3.0 * np.exp(-epochs / 30) + 0.5
        unstable_loss[40:60] += np.linspace(0, 1.5, 20)  # Spike
        unstable_loss += np.random.randn(100) * 0.1
        
        # Without gradient clipping (explosion)
        explosion_loss = 3.0 * np.exp(-epochs / 25) + 0.5
        explosion_loss[30] = 8.0  # Gradient explosion
        explosion_loss += np.random.randn(100) * 0.08
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, ideal_loss, 'g-', label='Ideal (all techniques)', linewidth=2)
        ax.plot(epochs, unstable_loss, 'orange', label='No Pre-norm (unstable)', linewidth=2)
        ax.plot(epochs, explosion_loss, 'r-', label='No gradient clipping (explosion)', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Stability: Impact of Architecture Choices')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 9)
        
        plt.tight_layout()
        plt.savefig('training_stability.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'training_stability.png'")
        print()
        
        print("OBSERVATIONS:")
        print("  • Ideal: Smooth, steady decrease")
        print("  • No Pre-norm: Oscillations and spikes")
        print("  • No clipping: Can completely fail (loss explosion)")
        print()


def main():
    """Main execution for Week 8."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "WEEK 8: NORMALIZATION" + " " * 27 + "║")
    print("║" + " " * 15 + "RMSNorm, Pre-norm & Stability" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Normalization comparison
    norm = NormalizationComparison()
    norm.compare_normalizations()
    
    # Architecture comparison
    arch = ArchitectureComparison()
    arch.explain_architectures()
    arch.visualize_gradient_flow()
    arch.compare_architecture_choices()
    
    # Stability techniques
    stability = TrainingStabilityTechniques()
    stability.explain_stability_techniques()
    stability.visualize_stability_comparison()
    
    print("=" * 70)
    print("🎯 WEEK 8 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. LAYER NORM vs RMS NORM:")
    print("   • LayerNorm: (x - mean) / std (centers and scales)")
    print("   • RMSNorm: x / RMS (just scales, faster)")
    print("   • RMSNorm is ~30-40% faster with similar performance")
    print("   • Modern choice: RMSNorm")
    print()
    print("2. PRE-NORM vs POST-NORM:")
    print("   • Post-norm: Norm → Sublayer → Residual (unstable)")
    print("   • Pre-norm: Sublayer → Residual → Norm (stable)")
    print("   • Pre-norm enables training deeper networks")
    print("   • Modern choice: Pre-norm")
    print()
    print("3. THE MODERN STACK (2024):")
    print("   • RMSNorm + Pre-norm")
    print("   • SwiGLU activation")
    print("   • GQA attention")
    print("   • RoPE position encoding")
    print()
    print("4. STABILITY TECHNIQUES:")
    print("   • Residual connections: Direct gradient flow")
    print("   • Pre-norm: Stable activations")
    print("   • Gradient clipping: Prevents explosion")
    print("   • LR warmup: Smooth start")
    print()
    print("5. YOUR UPGRADE PATH:")
    print("   • Replace LayerNorm with RMSNorm in layers.py")
    print("   • Switch from Post-norm to Pre-norm")
    print("   • Verify gradient clipping is enabled")
    print()
    print("=" * 70)
    print()
    print("Next: Week 9 - Modern Training Recipes")
    print("      (Chinchilla scaling laws, compute-optimal training)")
    print()


if __name__ == "__main__":
    main()
