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

"""Week 9: Modern Training Recipes - Chinchilla Scaling Laws.

This script demonstrates:
1. Chinchilla scaling laws (how to scale model vs data optimally)
2. Compute-optimal training
3. Parameter efficiency
4. Why most models are undertrained

Key paper: "Training Compute-Optimal Large Language Models" (Hoffmann et al. 2022)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')


class ChinchillaScaling:
    """Demonstrates Chinchilla scaling laws."""
    
    def __init__(self):
        """Initialize."""
        pass
    
    def explain_chinchilla(self) -> None:
        """Explain Chinchilla scaling laws."""
        print("=" * 70)
        print("🐿️  CHINCHILLA SCALING LAWS")
        print("=" * 70)
        print()
        
        print("THE PROBLEM:")
        print("  GPT-3 (2020): 175B params, 300B tokens")
        print("  Ratio: ~1,700 tokens per parameter")
        print()
        print("  But what if we trained longer?")
        print("  What if we used a smaller model with more data?")
        print()
        
        print("THE CHINCHILLA PAPER (2022):")
        print("  'Training Compute-Optimal Large Language Models'")
        print("  Authors: Hoffmann et al. (DeepMind)")
        print()
        print("  Key finding: GPT-3 was UNDERTRAINED!")
        print()
        
        print("THE SCALING LAWS:")
        print()
        print("  1. LOSS ∝ C^(-0.05)  where C = compute (FLOPs)")
        print()
        print("  2. OPTIMAL TOKENS/PARAMETER = 20")
        print("     (Not 1,700 like GPT-3!)")
        print()
        print("  3. For a given compute budget, there's an optimal")
        print("     trade-off between model size and training tokens")
        print()
        
        print("PRACTICAL IMPLICATIONS:")
        print()
        print("  For 175B model:")
        print("    GPT-3 way:  175B params × 300B tokens = Undertrained")
        print("    Chinchilla: 175B params × 3.5T tokens = Compute-optimal")
        print()
        print("  For same compute budget:")
        print("    Better: 70B params × 1.4T tokens = Same compute, better performance!")
        print()
    
    def calculate_optimal_tokens(self, params: float) -> float:
        """Calculate optimal number of tokens for given parameters.
        
        Chinchilla rule: tokens = 20 × params
        """
        return 20 * params
    
    def calculate_compute_budget(self, params: float, tokens: float) -> float:
        """Calculate compute budget in FLOPs.
        
        Roughly: C ≈ 6 × params × tokens
        (6 = 2 for forward + 4 for backward)
        """
        return 6 * params * tokens
    
    def visualize_scaling_laws(self) -> None:
        """Visualize Chinchilla scaling laws."""
        print("=" * 70)
        print("📊 CHINCHILLA SCALING VISUALIZATION")
        print("=" * 70)
        print()
        
        # Model sizes (in billions)
        model_sizes = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 175, 280, 540]
        
        # Calculate optimal tokens and compute for each
        data = []
        for params_b in model_sizes:
            params = params_b * 1e9
            optimal_tokens = self.calculate_optimal_tokens(params)
            compute = self.calculate_compute_budget(params, optimal_tokens)
            
            data.append({
                'params_b': params_b,
                'params': params,
                'tokens': optimal_tokens,
                'tokens_b': optimal_tokens / 1e9,
                'compute': compute,
                'compute_e': compute / 1e18  # ExaFLOPs
            })
        
        # Print table
        print("OPTIMAL TRAINING CONFIGURATIONS:")
        print()
        print(f"{'Params (B)':<12} {'Tokens (B)':<12} {'Ratio':<8} {'Compute (E)':<12}")
        print("-" * 50)
        for d in data:
            print(f"{d['params_b']:<12.1f} {d['tokens_b']:<12.1f} "
                  f"{20:<8} {d['compute_e']:<12.1f}")
        print()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Optimal tokens vs parameters
        ax1 = axes[0, 0]
        params_list = [d['params_b'] for d in data]
        tokens_list = [d['tokens_b'] for d in data]
        
        ax1.plot(params_list, tokens_list, 'b-o', linewidth=2, markersize=6)
        ax1.plot(params_list, [20*p for p in params_list], 'r--', 
                linewidth=2, label='tokens = 20 × params')
        
        # Mark famous models
        famous_models = {
            'GPT-3': (175, 300, 'red'),
            'GPT-4': (1750, 13000, 'purple'),
            'Chinchilla': (70, 1400, 'green'),
        }
        
        for name, (p, t, color) in famous_models.items():
            ax1.scatter([p], [t], color=color, s=100, zorder=5)
            ax1.annotate(name, (p, t), xytext=(10, 10), 
                        textcoords='offset points', fontweight='bold')
        
        ax1.set_xlabel('Model Parameters (Billions)')
        ax1.set_ylabel('Optimal Training Tokens (Billions)')
        ax1.set_title('Chinchilla Scaling Law\nOptimal Tokens vs Model Size')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        
        # Plot 2: Compute vs performance trade-off
        ax2 = axes[0, 1]
        
        # For fixed compute budget, show different model sizes
        fixed_compute = 1e21  # 1000 ExaFLOPs
        
        # Model sizes that fit this compute budget
        test_params = np.linspace(1e9, 200e9, 100)
        test_tokens = fixed_compute / (6 * test_params)
        
        # Estimate loss (simplified scaling law)
        # Loss ∝ (params)^(-0.34) and (tokens)^(-0.28)
        # Combined: Loss ∝ (compute)^(-0.05)
        losses = 3.0 * (fixed_compute)**(-0.05) * (test_params / 70e9)**0.1
        
        ax2.plot(test_params / 1e9, losses, 'b-', linewidth=2)
        
        # Mark optimal point
        optimal_params = (fixed_compute / (6 * 20))**0.5 / 1e9
        optimal_loss = 3.0 * (fixed_compute)**(-0.05)
        ax2.scatter([optimal_params], [optimal_loss], color='red', s=200, zorder=5)
        ax2.annotate('Optimal\n(Chinchilla)', 
                    (optimal_params, optimal_loss),
                    xytext=(optimal_params+20, optimal_loss+0.1),
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        ax2.set_xlabel('Model Size (Billions)')
        ax2.set_ylabel('Estimated Loss (Lower is Better)')
        ax2.set_title(f'Fixed Compute Budget: {fixed_compute/1e18:.0f} ExaFLOPs\n'
                     f'Smaller model + more data = Better!')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: GPT-3 vs Chinchilla comparison
        ax3 = axes[1, 0]
        
        scenarios = [
            {'name': 'GPT-3\n(2020)', 'params': 175, 'tokens': 300, 
             'compute': 6*175*300, 'efficiency': 0.7},
            {'name': 'Chinchilla\n(2022)', 'params': 70, 'tokens': 1400, 
             'compute': 6*70*1400, 'efficiency': 1.0},
            {'name': 'Chinchilla-70B\n(Optimal)', 'params': 70, 'tokens': 1400, 
             'compute': 6*70*1400, 'efficiency': 1.0},
        ]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        params_vals = [s['params'] for s in scenarios]
        tokens_vals = [s['tokens'] for s in scenarios]
        
        bars1 = ax3.bar(x - width/2, params_vals, width, label='Params (B)', alpha=0.7)
        bars2 = ax3.bar(x + width/2, tokens_vals, width, label='Tokens (B)', alpha=0.7)
        
        ax3.set_ylabel('Billions')
        ax3.set_title('GPT-3 vs Chinchilla Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s['name'] for s in scenarios])
        ax3.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.0f}B', ha='center', va='bottom', fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.0f}B', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Efficiency frontier
        ax4 = axes[1, 1]
        
        # Generate pareto frontier
        param_range = np.logspace(0, 3, 50)  # 1B to 1000B
        optimal_tokens = 20 * param_range
        compute_range = 6 * param_range * optimal_tokens
        
        # Plot efficiency frontier
        ax4.plot(param_range, optimal_tokens, 'g-', linewidth=3, 
                label='Chinchilla Optimal (20:1)')
        
        # Plot actual models
        actual_models = [
            ('GPT-3', 175, 300, 'red'),
            ('LaMDA', 137, 168, 'orange'),
            ('Chinchilla', 70, 1400, 'green'),
            ('PaLM', 540, 780, 'purple'),
            ('LLaMA-65B', 65, 1024, 'blue'),
            ('LLaMA-7B', 7, 1024, 'cyan'),
        ]
        
        for name, p, t, color in actual_models:
            ax4.scatter([p], [t], color=color, s=150, zorder=5)
            ax4.annotate(name, (p, t), xytext=(5, 5),
                        textcoords='offset points', fontsize=9)
        
        # Add ratio lines
        for ratio in [1, 10, 20, 100]:
            line_params = np.logspace(0, 3, 50)
            line_tokens = ratio * line_params
            ax4.plot(line_params, line_tokens, 'k--', alpha=0.2, linewidth=0.5)
            ax4.text(line_params[-1], line_tokens[-1], f' {ratio}:1',
                   fontsize=8, alpha=0.5)
        
        ax4.set_xlabel('Model Parameters (Billions) - Log Scale')
        ax4.set_ylabel('Training Tokens (Billions) - Log Scale')
        ax4.set_title('Chinchilla Efficiency Frontier\n(Green = Optimal)')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('chinchilla_scaling.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'chinchilla_scaling.png'")
        print()
    
    def analyze_real_models(self) -> None:
        """Analyze real models with Chinchilla lens."""
        print("=" * 70)
        print("🔍 REAL MODEL ANALYSIS")
        print("=" * 70)
        print()
        
        models = [
            {'name': 'GPT-3', 'year': 2020, 'params': 175, 'tokens': 300},
            {'name': 'GPT-4', 'year': 2023, 'params': 1760, 'tokens': 13000},
            {'name': 'LaMDA', 'year': 2022, 'params': 137, 'tokens': 168},
            {'name': 'PaLM', 'year': 2022, 'params': 540, 'tokens': 780},
            {'name': 'Chinchilla', 'year': 2022, 'params': 70, 'tokens': 1400},
            {'name': 'LLaMA-7B', 'year': 2023, 'params': 7, 'tokens': 1024},
            {'name': 'LLaMA-65B', 'year': 2023, 'params': 65, 'tokens': 1024},
            {'name': 'Mistral-7B', 'year': 2023, 'params': 7, 'tokens': 7000},
            {'name': 'Llama 3 8B', 'year': 2024, 'params': 8, 'tokens': 15000},
            {'name': 'Llama 3 70B', 'year': 2024, 'params': 70, 'tokens': 15000},
        ]
        
        print(f"{'Model':<15} {'Params':<8} {'Tokens':<8} {'Ratio':<8} {'Status':<15}")
        print("-" * 65)
        
        for m in models:
            ratio = m['tokens'] / m['params']
            
            if ratio < 10:
                status = "Undertrained"
            elif ratio < 20:
                status = "Slightly under"
            elif ratio < 30:
                status = "✅ Optimal"
            else:
                status = "Overtrained"
            
            print(f"{m['name']:<15} {m['params']:<8.0f} {m['tokens']:<8.0f} "
                  f"{ratio:<8.1f} {status:<15}")
        
        print()
        print("KEY INSIGHTS:")
        print("  • GPT-3: Very undertrained (ratio 1.7)")
        print("  • LLaMA-65B: Near optimal (ratio 15.8)")
        print("  • LLaMA-7B: Overtrained (ratio 146)")
        print("  • Llama 3: Well trained (ratio 214)")
        print("  • Modern trend: Smaller models, more data!")
        print()


class TrainingRecipes:
    """Modern training recipes and best practices."""
    
    def explain_training_recipes(self) -> None:
        """Explain modern training recipes."""
        print("=" * 70)
        print("📜 MODERN TRAINING RECIPES")
        print("=" * 70)
        print()
        
        print("RECIPE 1: Chinchilla-Optimal (70B model)")
        print("  Parameters: 70B")
        print("  Tokens: 1.4T")
        print("  Ratio: 20:1")
        print("  Compute: ~600 ExaFLOPs")
        print("  Estimated cost: ~$1M")
        print()
        
        print("RECIPE 2: Small but Mighty (7B model)")
        print("  Parameters: 7B")
        print("  Tokens: 2T (overtrained, but high quality)")
        print("  Ratio: 286:1")
        print("  Compute: ~84 ExaFLOPs")
        print("  Estimated cost: ~$100K")
        print("  Result: Mistral-7B quality (beats larger models!)")
        print()
        
        print("RECIPE 3: Compute-Limited (1B model)")
        print("  Parameters: 1B")
        print("  Tokens: 20B (optimal)")
        print("  Ratio: 20:1")
        print("  Compute: ~120 PetaFLOPs")
        print("  Estimated cost: ~$5K")
        print("  Good for: Research, prototyping, edge deployment")
        print()
        
        print("HYPERPARAMETERS (Modern Standard):")
        print()
        print("  Learning Rate:")
        print("    • Schedule: Warmup (1%) + Cosine decay")
        print("    • Base LR: 1e-4 to 3e-4 (depends on model size)")
        print("    • Final LR: ~10% of base")
        print()
        print("  Optimizer:")
        print("    • AdamW (always)")
        print("    • β1=0.9, β2=0.95 (not 0.999)")
        print("    • Weight decay: 0.1")
        print()
        print("  Batch Size:")
        print("    • Global: 4M tokens (modern standard)")
        print("    • Per device: As large as fits")
        print("    • Gradient accumulation: Use if needed")
        print()
        print("  Training Stability:")
        print("    • Gradient clipping: 1.0")
        print("    • Mixed precision: BF16 (preferred) or FP16")
        print("    • Loss spike recovery: Skip bad batches")
        print()
    
    def hyperparameter_scaling(self) -> None:
        """Explain how hyperparameters scale with model size."""
        print("=" * 70)
        print("⚙️  HYPERPARAMETER SCALING")
        print("=" * 70)
        print()
        
        print("LEARNING RATE SCALING:")
        print()
        print("  Smaller models → Larger learning rates")
        print("  Larger models → Smaller learning rates")
        print()
        print("  Rules of thumb:")
        print("    • 1B model:   LR ~ 3e-4")
        print("    • 7B model:   LR ~ 3e-4")
        print("    • 70B model:  LR ~ 1.5e-4")
        print("    • 175B model: LR ~ 1e-4")
        print()
        print("  Formula (approximate):")
        print("    LR ∝ 1 / ∛(model_params)")
        print()
        
        print("BATCH SIZE SCALING:")
        print()
        print("  Modern standard: ~4M tokens per batch")
        print()
        print("  For different model sizes:")
        print("    • 1B model:  4M tokens, 512 seq_len → batch ~ 8K sequences")
        print("    • 7B model:  4M tokens → batch ~ 8K sequences")
        print("    • 70B model: 4M tokens → batch ~ 8K sequences")
        print()
        print("  Why constant? Evidence shows it works well across sizes")
        print()
        
        print("WARMUP SCALING:")
        print()
        print("  Standard: 1-2% of total training steps")
        print()
        print("  Examples:")
        print("    • 100B tokens @ 4M batch → 25K steps → 250-500 warmup steps")
        print("    • 1T tokens @ 4M batch → 250K steps → 2.5K-5K warmup steps")
        print()
    
    def common_mistakes(self) -> None:
        """Explain common training mistakes."""
        print("=" * 70)
        print("❌ COMMON TRAINING MISTAKES")
        print("=" * 70)
        print()
        
        mistakes = [
            {
                'mistake': 'Training too short',
                'example': '70B model on 100B tokens (ratio 1.4)',
                'problem': 'Model severely undertrained',
                'fix': 'Use Chinchilla ratio: 1.4T tokens',
            },
            {
                'mistake': 'Learning rate too high',
                'example': 'LR = 1e-3 for 70B model',
                'problem': 'Training unstable, loss spikes',
                'fix': 'Reduce to 1.5e-4',
            },
            {
                'mistake': 'No learning rate decay',
                'example': 'Constant LR throughout',
                'problem': 'Cannot converge to sharp minimum',
                'fix': 'Use cosine decay to 10% of base',
            },
            {
                'mistake': 'Wrong β2 for Adam',
                'example': 'β2 = 0.999 (default)',
                'problem': 'Slow adaptation, poor convergence',
                'fix': 'Use β2 = 0.95 (modern standard)',
            },
            {
                'mistake': 'Gradient accumulation only',
                'example': 'Effective batch = 4M but per-device = 8',
                'problem': 'Too many accumulation steps (slow!)',
                'fix': 'Increase per-device batch or use more GPUs',
            },
            {
                'mistake': 'Ignoring loss spikes',
                'example': 'Continue training after spike',
                'problem': 'Model may never recover',
                'fix': 'Rollback to checkpoint before spike',
            },
        ]
        
        for i, m in enumerate(mistakes, 1):
            print(f"{i}. {m['mistake'].upper()}")
            print(f"   Example: {m['example']}")
            print(f"   Problem: {m['problem']}")
            print(f"   Fix: {m['fix']}")
            print()


def main():
    """Main execution for Week 9."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "WEEK 9: MODERN TRAINING RECIPES" + " " * 17 + "║")
    print("║" + " " * 18 + "Chinchilla Scaling Laws" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Chinchilla scaling
    chinchilla = ChinchillaScaling()
    chinchilla.explain_chinchilla()
    chinchilla.visualize_scaling_laws()
    chinchilla.analyze_real_models()
    
    # Training recipes
    recipes = TrainingRecipes()
    recipes.explain_training_recipes()
    recipes.hyperparameter_scaling()
    recipes.common_mistakes()
    
    print("=" * 70)
    print("🎯 WEEK 9 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. CHINCHILLA SCALING LAW:")
    print("   • Optimal tokens = 20 × parameters")
    print("   • GPT-3 was UNDERTRAINED (1.7K vs 20 ratio)")
    print("   • Smaller model + more data often beats larger model")
    print()
    print("2. COMPUTE-OPTIMAL TRAINING:")
    print("   • For fixed compute, use smaller model + more tokens")
    print("   • Example: 70B + 1.4T beats 175B + 300B")
    print()
    print("3. MODERN HYPERPARAMETERS:")
    print("   • LR: 1e-4 to 3e-4 (scale down with model size)")
    print("   • β1=0.9, β2=0.95 (not default 0.999)")
    print("   • Batch: ~4M tokens (constant across sizes)")
    print("   • Warmup: 1-2% of total steps")
    print()
    print("4. MODERN TREND (2024):")
    print("   • Smaller models (7B-70B)")
    print("   • MUCH more data (1T-15T tokens)")
    print("   • High quality curated data")
    print("   • Longer training runs")
    print()
    print("5. PRACTICAL RECIPES:")
    print("   • Small model (7B): 2T tokens, ~$100K")
    print("   • Medium model (70B): 1.4T tokens, ~$1M")
    print("   • Research model (1B): 20B tokens, ~$5K")
    print()
    print("=" * 70)
    print()
    print("Next: Week 10 - Post-Training (RLHF, DPO)")
    print("      Or plan your own training run with Chinchilla rules!")
    print()


if __name__ == "__main__":
    main()
