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

"""Week 6: Distributed Training & Model Parallelism.

This script demonstrates concepts for training across multiple GPUs:
1. Data parallelism (same model, different data)
2. Model parallelism (split model across GPUs)
3. Pipeline parallelism (layer-wise distribution)
4. Communication and synchronization

Note: This is a conceptual simulation since we run on single GPU.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time


class DistributedTrainingConcepts:
    """Demonstrates distributed training concepts."""
    
    def explain_data_parallelism(self) -> None:
        """Explain data parallelism concept."""
        print("=" * 70)
        print("📊 DATA PARALLELISM (DP)")
        print("=" * 70)
        print()
        
        print("THE CONCEPT:")
        print("  Each GPU has the FULL model")
        print("  Each GPU processes a DIFFERENT batch")
        print("  Gradients are averaged across GPUs")
        print("  All GPUs update the same model")
        print()
        
        print("VISUALIZATION:")
        print("  ┌──────────────────────────────────────────────────────┐")
        print("  │                  DATA PARALLELISM                    │")
        print("  ├──────────────────────────────────────────────────────┤")
        print("  │                                                      │")
        print("  │  ┌─────────┐    ┌─────────┐    ┌─────────┐        │")
        print("  │  │ Batch 0 │    │ Batch 1 │    │ Batch 2 │        │")
        print("  │  │  GPU 0  │    │  GPU 1  │    │  GPU 2  │        │")
        print("  │  │         │    │         │    │         │        │")
        print("  │  │ ┌───┐   │    │ ┌───┐   │    │ ┌───┐   │        │")
        print("  │  │ │ M │   │    │ │ M │   │    │ │ M │   │        │")
        print("  │  │ │ o │   │    │ │ o │   │    │ │ o │   │        │")
        print("  │  │ │ d │   │    │ │ d │   │    │ │ d │   │        │")
        print("  │  │ │ e │   │    │ │ e │   │    │ │ e │   │        │")
        print("  │  │ │ l │   │    │ │ l │   │    │ │ l │   │        │")
        print("  │  │ └───┘   │    │ └───┘   │    │ └───┘   │        │")
        print("  │  │         │    │         │    │         │        │")
        print("  │  │ Grad 0  │    │ Grad 1  │    │ Grad 2  │        │")
        print("  │  └────┬────┘    └────┬────┘    └────┬────┘        │")
        print("  │       │              │              │               │")
        print("  │       └──────────────┼──────────────┘               │")
        print("  │                      ↓                             │")
        print("  │            ┌─────────────────┐                     │")
        print("  │            │  Average      │                     │")
        print("  │            │  Gradients    │                     │")
        print("  │            └───────┬─────────┘                     │")
        print("  │                    ↓                               │")
        print("  │            ┌───────────────┐                       │")
        print("  │            │ Update Model  │                       │")
        print("  │            │ (All GPUs)    │                       │")
        print("  │            └───────────────┘                       │")
        print("  │                                                      │")
        print("  └──────────────────────────────────────────────────────┘")
        print()
        
        print("CHARACTERISTICS:")
        print("  ✅ Simple to implement")
        print("  ✅ Near-linear speedup (if batch size large enough)")
        print("  ❌ Each GPU needs full model (memory limit)")
        print("  ❌ Communication overhead (all-reduce gradients)")
        print()
        
        print("WHEN TO USE:")
        print("  • Model fits on single GPU")
        print("  • Want to train on larger batches")
        print("  • Most common form of parallelism")
        print()
        
        print("COMMUNICATION:")
        print("  Each epoch requires ALL-REDUCE:")
        print("    GPU 0 grad: [0.5, -0.2, 0.1, ...]")
        print("    GPU 1 grad: [0.4, -0.1, 0.2, ...]")
        print("    GPU 2 grad: [0.6, -0.3, 0.1, ...]")
        print("    ─────────────────────────────────")
        print("    Avg:        [0.5, -0.2, 0.13, ...] ← All GPUs get this")
        print()
    
    def explain_model_parallelism(self) -> None:
        """Explain model parallelism concept."""
        print("=" * 70)
        print("🏗️  MODEL PARALLELISM (MP)")
        print("=" * 70)
        print()
        
        print("THE PROBLEM:")
        print("  Model is too big to fit on single GPU")
        print("  Example: GPT-3 175B = 700GB, GPU has 80GB")
        print()
        
        print("THE SOLUTION:")
        print("  Split model layers across GPUs")
        print("  Each GPU holds part of the model")
        print("  Data flows through GPUs sequentially")
        print()
        
        print("VISUALIZATION:")
        print("  ┌──────────────────────────────────────────────────────┐")
        print("  │                 MODEL PARALLELISM                    │")
        print("  ├──────────────────────────────────────────────────────┤")
        print("  │                                                      │")
        print("  │  Input: [batch, seq_len, vocab]                    │")
        print("  │                      ↓                               │")
        print("  │  ┌──────────────────────────────────────────────┐   │")
        print("  │  │ GPU 0: Layers 0-7 (Embedding + 8 blocks)     │   │")
        print("  │  │ Size: 80GB                                   │   │")
        print("  │  └──────────────────────┬───────────────────────┘   │")
        print("  │                         ↓                            │")
        print("  │  ┌──────────────────────────────────────────────┐   │")
        print("  │  │ GPU 1: Layers 8-15 (8 blocks)                │   │")
        print("  │  │ Size: 80GB                                   │   │")
        print("  │  └──────────────────────┬───────────────────────┘   │")
        print("  │                         ↓                            │")
        print("  │  ┌──────────────────────────────────────────────┐   │")
        print("  │  │ GPU 2: Layers 16-23 (8 blocks)               │   │")
        print("  │  │ Size: 80GB                                   │   │")
        print("  │  └──────────────────────┬───────────────────────┘   │")
        print("  │                         ↓                            │")
        print("  │  ┌──────────────────────────────────────────────┐   │")
        print("  │  │ GPU 3: Layers 24-31 + Output (8 blocks + head)│   │")
        print("  │  │ Size: 80GB                                   │   │")
        print("  │  └──────────────────────┬───────────────────────┘   │")
        print("  │                         ↓                            │")
        print("  │                 Output logits                        │")
        print("  │                                                      │")
        print("  └──────────────────────────────────────────────────────┘")
        print()
        
        print("CHARACTERISTICS:")
        print("  ✅ Can train models larger than single GPU")
        print("  ❌ Underutilization (only one GPU active at a time)")
        print("  ❌ High communication (activations passed between GPUs)")
        print("  ❌ Complex to implement")
        print()
        
        print("WHEN TO USE:")
        print("  • Model doesn't fit on single GPU")
        print("  • Not the first choice (try data parallelism first)")
        print()
    
    def explain_pipeline_parallelism(self) -> None:
        """Explain pipeline parallelism."""
        print("=" * 70)
        print("🔀 PIPELINE PARALLELISM (PP)")
        print("=" * 70)
        print()
        
        print("THE PROBLEM:")
        print("  Model parallelism has low GPU utilization")
        print("  Only 1 GPU active at a time!")
        print()
        
        print("THE SOLUTION:")
        print("  Split model into stages (like model parallelism)")
        print("  Process multiple mini-batches concurrently")
        print("  Pipeline the forward/backward passes")
        print()
        
        print("VISUALIZATION (Micro-batching):")
        print("  Time →")
        print()
        print("  GPU 0: [F0][F1][F2][F3][B3][B2][B1][B0]")
        print("  GPU 1: ____[F0][F1][F2][F3][B3][B2][B1][B0]")
        print("  GPU 2: ________[F0][F1][F2][F3][B3][B2][B1][B0]")
        print("  GPU 3: ____________[F0][F1][F2][F3][B3][B2][B1][B0]")
        print()
        print("  F = Forward pass on micro-batch")
        print("  B = Backward pass on micro-batch")
        print()
        
        print("RESULT: All GPUs active most of the time!")
        print()
        
        print("THE 'BUBBLE' PROBLEM:")
        print("  At start and end of batch, some GPUs are idle")
        print("  More micro-batches = smaller bubble")
        print()
        print("  With 4 micro-batches:")
        print("  ════════════════════════════════════════════════════")
        print("  GPU 0: F0 F1 F2 F3 B3 B2 B1 B0")
        print("  GPU 1: __ F0 F1 F2 F3 B3 B2 B1 B0")
        print("  GPU 2: ____ F0 F1 F2 F3 B3 B2 B1 B0")
        print("  GPU 3: ______ F0 F1 F2 F3 B3 B2 B1 B0")
        print("  ════════════════════════════════════════════════════")
        print("        ↑    ↑             ↑    ↑")
        print("     Bubble           Bubble")
        print()
        
        print("With 32 micro-batches, bubble is only 12.5% of time!")
        print()
    
    def compare_parallelism_strategies(self) -> None:
        """Compare different parallelism strategies."""
        print("=" * 70)
        print("⚖️  PARALLELISM STRATEGY COMPARISON")
        print("=" * 70)
        print()
        
        print("┌──────────────────┬───────────────┬───────────────┬───────────────┐")
        print("│ Aspect           │ Data Parallel │ Model Parallel│ Pipeline      │")
        print("├──────────────────┼───────────────┼───────────────┼───────────────┤")
        print("│ Model per GPU    │ Full model    │ Part of model │ Part of model │")
        print("│ Data per GPU     │ Different     │ Same          │ Same          │")
        print("│ GPU Utilization  │ High          │ Low           │ High (with PP)│")
        print("│ Communication    │ Gradients     │ Activations   │ Activations   │")
        print("│ Complexity       │ Low           │ High          │ Medium        │")
        print("│ Use Case         │ Model fits    │ Model too big │ Model too big │")
        print("│                  │ in one GPU    │ for one GPU   │ for one GPU   │")
        print("└──────────────────┴───────────────┴───────────────┴───────────────┘")
        print()
        
        print("MODERN APPROACH: 3D PARALLELISM")
        print("  Combine all three strategies!")
        print()
        print("  Example: Training GPT-3 on 1024 GPUs")
        print("  • Data parallelism: 16 replicas (16× batch size)")
        print("  • Pipeline parallelism: 4 stages per replica")
        print("  • Tensor parallelism: 16 GPUs per stage")
        print("  • Total: 16 × 4 × 16 = 1024 GPUs")
        print()
        print("  Result: Can train models with trillions of parameters!")
        print()
    
    def simulate_scaling_efficiency(self) -> None:
        """Simulate scaling efficiency."""
        print("=" * 70)
        print("📈 SCALING EFFICIENCY")
        print("=" * 70)
        print()
        
        print("IDEAL: Linear Speedup")
        print("  2 GPUs → 2× faster")
        print("  4 GPUs → 4× faster")
        print("  8 GPUs → 8× faster")
        print()
        
        print("REALITY: Communication overhead")
        print("  Each GPU must wait for others")
        print("  Network bandwidth is the bottleneck")
        print()
        
        # Simulate efficiency
        num_gpus = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        
        # Ideal speedup
        ideal_speedup = num_gpus
        
        # Realistic speedup (with communication overhead)
        # Efficiency drops as we add more GPUs
        efficiency = 1.0 - 0.02 * np.log2(num_gpus)  # 2% loss per doubling
        efficiency = np.clip(efficiency, 0.5, 1.0)  # Floor at 50%
        real_speedup = num_gpus * efficiency
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Speedup
        ax1.plot(num_gpus, ideal_speedup, 'g--', label='Ideal (linear)', linewidth=2)
        ax1.plot(num_gpus, real_speedup, 'b-o', label='Real (with overhead)', linewidth=2)
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Scaling Efficiency: Ideal vs Real')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log', base=2)
        
        # Efficiency percentage
        efficiency_pct = (real_speedup / num_gpus) * 100
        ax2.plot(num_gpus, efficiency_pct, 'r-o', linewidth=2)
        ax2.set_xlabel('Number of GPUs')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Scaling Efficiency Percentage')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        for i, (n, e) in enumerate(zip(num_gpus, efficiency_pct)):
            if i % 2 == 0:
                ax2.annotate(f'{e:.0f}%', xy=(n, e), 
                           xytext=(n, e+3), ha='center')
        
        plt.tight_layout()
        plt.savefig('scaling_efficiency.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'scaling_efficiency.png'")
        print()
        
        print("RESULTS:")
        for n, ideal, real, eff in zip(num_gpus, ideal_speedup, real_speedup, efficiency_pct):
            print(f"  {n:3d} GPUs: {real:6.1f}× speedup (ideal: {ideal:3d}×, {eff:4.1f}% efficient)")
        print()
        
        print("KEY INSIGHT:")
        print("  • 2-8 GPUs: Great efficiency (>90%)")
        print("  • 32+ GPUs: Diminishing returns (<80%)")
        print("  • 128+ GPUs: Need careful optimization")
        print()


class ZeROExplanation:
    """Explain ZeRO optimizer (memory optimization)."""
    
    def explain_zero(self) -> None:
        """Explain ZeRO stages."""
        print("=" * 70)
        print("🔄 ZERO: ZERO REDUNDANCY OPTIMIZER")
        print("=" * 70)
        print()
        
        print("THE PROBLEM:")
        print("  Adam optimizer stores 2× model size (momentum + variance)")
        print("  With data parallelism, EACH GPU has full optimizer state!")
        print("  Wasteful: All GPUs have identical copies")
        print()
        
        print("THE SOLUTION (ZeRO):")
        print("  Partition optimizer states across GPUs")
        print("  Each GPU only stores part of the state")
        print("  Gather when needed for updates")
        print()
        
        print("ZER0 STAGES:")
        print()
        
        print("Stage 1: Partition Optimizer States")
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  Before: 4 GPUs × 2× model = 8× model total           │")
        print("  │  After:  Each GPU has 0.5× model optimizer state     │")
        print("  │  Memory: 4× model (was 8×) - 50% reduction!          │")
        print("  └────────────────────────────────────────────────────────┘")
        print()
        
        print("Stage 2: + Partition Gradients")
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  Each GPU only computes gradients for its parameters │")
        print("  │  Memory: 2× model (was 8×) - 75% reduction!          │")
        print("  └────────────────────────────────────────────────────────┘")
        print()
        
        print("Stage 3: + Partition Parameters")
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  Each GPU only holds its parameters                  │")
        print("  │  Gather parameters when needed for forward/backward    │")
        print("  │  Memory: 1× model (was 8×) - 87.5% reduction!        │")
        print("  └────────────────────────────────────────────────────────┘")
        print()
        
        print("VISUALIZATION (4 GPUs, model = 10GB):")
        print()
        print("┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│ Component   │ No ZeRO  │ ZeRO-1   │ ZeRO-2   │ ZeRO-3   │")
        print("├─────────────┼──────────┼──────────┼──────────┼──────────┤")
        print("│ Parameters  │ 10 GB    │ 10 GB    │ 10 GB    │ 2.5 GB   │")
        print("│ Gradients   │ 10 GB    │ 10 GB    │ 2.5 GB   │ 2.5 GB   │")
        print("│ Optimizer   │ 20 GB    │ 5 GB     │ 5 GB     │ 5 GB     │")
        print("├─────────────┼──────────┼──────────┼──────────┼──────────┤")
        print("│ Per GPU     │ 40 GB    │ 25 GB    │ 17.5 GB  │ 10 GB    │")
        print("│ Total (4×)  │ 160 GB   │ 100 GB   │ 70 GB    │ 40 GB    │")
        print("│ Reduction   │ Baseline │ 37.5%    │ 56.3%    │ 75%      │")
        print("└─────────────┴──────────┴──────────┴──────────┴──────────┘")
        print()
        
        print("IN YOUR CODEBASE:")
        print("  • DeepSpeed: Microsoft's implementation")
        print("  • FSDP: PyTorch's Fully Sharded Data Parallel")
        print("  • Both implement ZeRO-3 style optimization")
        print()
        print("WHEN TO USE:")
        print("  • Always for models > 1B parameters")
        print("  • Can train models 8× larger on same hardware!")
        print()


def main():
    """Main execution for Week 6."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "WEEK 6: DISTRIBUTED TRAINING" + " " * 20 + "║")
    print("║" + " " * 18 + "Multi-GPU & Model Parallelism" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    dist = DistributedTrainingConcepts()
    
    dist.explain_data_parallelism()
    dist.explain_model_parallelism()
    dist.explain_pipeline_parallelism()
    dist.compare_parallelism_strategies()
    dist.simulate_scaling_efficiency()
    
    zero = ZeROExplanation()
    zero.explain_zero()
    
    print("=" * 70)
    print("🎯 WEEK 6 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. DATA PARALLELISM:")
    print("   • Same model on each GPU, different data")
    print("   • Most common, easiest to implement")
    print("   • Limit: Model must fit on single GPU")
    print()
    print("2. MODEL PARALLELISM:")
    print("   • Split model layers across GPUs")
    print("   • For models too big for one GPU")
    print("   • Poor utilization (only 1 GPU active)")
    print()
    print("3. PIPELINE PARALLELISM:")
    print("   • Model parallelism + micro-batching")
    print("   • Better GPU utilization")
    print("   • 'Bubble' overhead at start/end")
    print()
    print("4. ZER0 (MEMORY OPTIMIZATION):")
    print("   • Partition optimizer states across GPUs")
    print("   • 8× memory reduction with ZeRO-3!")
    print("   • Essential for models > 1B params")
    print()
    print("5. SCALING EFFICIENCY:")
    print("   • 2-8 GPUs: >90% efficient")
    print("   • 32+ GPUs: Need careful optimization")
    print("   • Network bandwidth is the bottleneck")
    print()
    print("6. MODERN APPROACH:")
    print("   • 3D Parallelism: DP + PP + TP combined")
    print("   • Train trillion-parameter models!")
    print()
    print("=" * 70)
    print()
    print("Next: Week 7 - Advanced Architecture (MQA, GQA, RoPE)")
    print("      Or experiment with distributed training frameworks!")
    print()


if __name__ == "__main__":
    main()
