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

"""Week 5: Memory Profiling & Compute Efficiency.

This script demonstrates:
1. Memory usage breakdown (where does GPU memory go?)
2. Gradient accumulation (simulate large batches)
3. Mixed precision training (FP16/BF16)
4. Batch size effects on training

Essential for training real LLMs efficiently.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax
import jax.numpy as jnp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys
import time

sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')

from transformers.layers import (
    TokenAndPositionEmbedding,
    TransformerBlock,
)


class MemoryProfiler:
    """Profiles memory usage of transformer models."""
    
    def __init__(self):
        """Initialize profiler."""
        self.results = []
    
    def calculate_model_memory(self, vocab_size: int, max_length: int,
                              embedding_dim: int, num_blocks: int,
                              num_heads: int) -> Dict[str, float]:
        """Calculate theoretical memory usage.
        
        Returns memory breakdown in MB.
        """
        print("=" * 70)
        print("📊 MODEL MEMORY ANALYSIS")
        print("=" * 70)
        print()
        
        print(f"Configuration:")
        print(f"  Vocabulary: {vocab_size:,}")
        print(f"  Max length: {max_length}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Transformer blocks: {num_blocks}")
        print(f"  Attention heads: {num_heads}")
        print()
        
        # Calculate parameter counts
        # Embedding
        embed_params = vocab_size * embedding_dim
        pos_embed_params = max_length * embedding_dim
        
        # Per transformer block
        # Attention: Q, K, V projections + output projection
        head_dim = embedding_dim // num_heads
        attn_params = 4 * (embedding_dim * embedding_dim)  # Q, K, V, Out
        
        # FFN: expand + project
        mlp_dim = embedding_dim * 4
        ffn_params = (embedding_dim * mlp_dim) + (mlp_dim * embedding_dim)
        
        # Layer norms (2 per block)
        layernorm_params = 2 * 2 * embedding_dim  # gamma + beta per LN
        
        block_params = attn_params + ffn_params + layernorm_params
        total_block_params = block_params * num_blocks
        
        # Output layer
        output_params = embedding_dim * vocab_size
        
        total_params = embed_params + pos_embed_params + total_block_params + output_params
        
        # Memory calculations (FP32 = 4 bytes per parameter)
        bytes_per_param = 4
        
        memory = {
            'embeddings': (embed_params + pos_embed_params) * bytes_per_param / (1024**2),
            'attention': attn_params * num_blocks * bytes_per_param / (1024**2),
            'ffn': ffn_params * num_blocks * bytes_per_param / (1024**2),
            'layernorm': layernorm_params * num_blocks * bytes_per_param / (1024**2),
            'output': output_params * bytes_per_param / (1024**2),
            'total_params': total_params,
            'model_memory': total_params * bytes_per_param / (1024**2),
        }
        
        print("PARAMETER COUNTS:")
        print(f"  Embeddings:     {embed_params + pos_embed_params:>12,} ({memory['embeddings']:.1f} MB)")
        print(f"  Attention:      {attn_params * num_blocks:>12,} ({memory['attention']:.1f} MB)")
        print(f"  FFN:            {ffn_params * num_blocks:>12,} ({memory['ffn']:.1f} MB)")
        print(f"  LayerNorm:      {layernorm_params * num_blocks:>12,} ({memory['layernorm']:.1f} MB)")
        print(f"  Output:         {output_params:>12,} ({memory['output']:.1f} MB)")
        print(f"  ─────────────────────────────────────────")
        print(f"  TOTAL:          {total_params:>12,} ({memory['model_memory']:.1f} MB)")
        print()
        
        # Training memory (much larger!)
        batch_size = 32
        print(f"TRAINING MEMORY (batch_size={batch_size}):")
        
        # Activations during forward pass
        seq_len = max_length
        
        # Embedding output: [batch, seq, dim]
        embed_activations = batch_size * seq_len * embedding_dim * bytes_per_param / (1024**2)
        
        # Per block activations (attention + FFN)
        # Attention: Q, K, V, attention scores, output
        attn_activations = batch_size * seq_len * (3 * embedding_dim + num_heads * seq_len + embedding_dim)
        attn_activations *= bytes_per_param / (1024**2)
        
        # FFN: hidden activations
        ffn_activations = batch_size * seq_len * (mlp_dim + embedding_dim)
        ffn_activations *= bytes_per_param / (1024**2)
        
        # Total per block
        block_activations = attn_activations + ffn_activations
        total_activations = embed_activations + block_activations * num_blocks
        
        # Gradients (same size as parameters)
        gradients_memory = memory['model_memory']
        
        # Optimizer states (Adam: 2x parameters for m and v)
        optimizer_memory = 2 * memory['model_memory']
        
        total_training = memory['model_memory'] + total_activations + gradients_memory + optimizer_memory
        
        print(f"  Model weights:     {memory['model_memory']:.1f} MB")
        print(f"  Activations:       {total_activations:.1f} MB")
        print(f"    - Embeddings:    {embed_activations:.1f} MB")
        print(f"    - Per block:     {block_activations:.1f} MB")
        print(f"    - Total blocks:  {block_activations * num_blocks:.1f} MB")
        print(f"  Gradients:         {gradients_memory:.1f} MB")
        print(f"  Optimizer states:  {optimizer_memory:.1f} MB (Adam: 2x model)")
        print(f"  ─────────────────────────────────────────")
        print(f"  TOTAL TRAINING:    {total_training:.1f} MB ({total_training/1024:.2f} GB)")
        print()
        
        print("💡 KEY INSIGHT:")
        print(f"   Training uses {total_training/memory['model_memory']:.1f}x more memory than model!")
        print(f"   Main culprits: Activations + Optimizer states")
        print()
        
        return memory
    
    def compare_model_sizes(self) -> None:
        """Compare memory for different model sizes."""
        print("=" * 70)
        print("📈 MODEL SIZE COMPARISON")
        print("=" * 70)
        print()
        
        configs = [
            {'name': 'Tiny (Demo)', 'vocab': 100, 'dim': 64, 'blocks': 2, 'heads': 4},
            {'name': 'Small', 'vocab': 1000, 'dim': 256, 'blocks': 4, 'heads': 8},
            {'name': 'Medium', 'vocab': 10000, 'dim': 512, 'blocks': 8, 'heads': 8},
            {'name': 'Large', 'vocab': 50000, 'dim': 768, 'blocks': 12, 'heads': 12},
            {'name': 'XL', 'vocab': 50000, 'dim': 1024, 'blocks': 24, 'heads': 16},
        ]
        
        print(f"{'Model':<12} {'Params':>12} {'Model MB':>10} {'Train GB':>10}")
        print("-" * 50)
        
        results = []
        for config in configs:
            vocab = config['vocab']
            dim = config['dim']
            blocks = config['blocks']
            heads = config['heads']
            
            # Estimate parameters
            embed = vocab * dim
            pos = 512 * dim  # Assume max 512
            
            # Per block
            head_dim = dim // heads
            attn = 4 * dim * dim
            ffn = dim * 4 * dim + 4 * dim * dim
            ln = 4 * dim
            block_params = attn + ffn + ln
            
            output = dim * vocab
            
            total = embed + pos + block_params * blocks + output
            model_mb = total * 4 / (1024**2)
            
            # Rough training estimate (model + 3x for activations/grads/optimizer)
            train_gb = model_mb * 4 / 1024
            
            print(f"{config['name']:<12} {total:>12,} {model_mb:>10.1f} {train_gb:>10.2f}")
            
            results.append({
                'name': config['name'],
                'params': total,
                'model_mb': model_mb,
                'train_gb': train_gb
            })
        
        print()
        print("For perspective:")
        print(f"  • GPT-2 Small:  117M params  (~470 MB model, ~2 GB training)")
        print(f"  • GPT-3:        175B params  (~700 GB model, HUGE training)")
        print(f"  • Your demos:    <1M params   (<10 MB model, <100 MB training)")
        print()


class GradientAccumulationDemo:
    """Demonstrates gradient accumulation for large batch simulation."""
    
    def __init__(self):
        """Initialize demo."""
        self.X, self.y = self._get_dataset()
    
    def _get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create dataset."""
        sequences = [
            ([1, 2, 3, 4, 1, 2, 3, 4], [2, 3, 4, 1, 2, 3, 4, 1]),
            ([5, 6, 5, 6, 5, 6, 5, 6], [6, 5, 6, 5, 6, 5, 6, 5]),
            ([7, 8, 9, 7, 8, 9, 7, 8], [8, 9, 7, 8, 9, 7, 8, 9]),
            ([10, 11, 10, 11, 10, 11, 10, 11], [11, 10, 11, 10, 11, 10, 11, 10]),
        ]
        X = np.array([s[0] for s in sequences] * 8)  # 32 samples
        y = np.array([s[1] for s in sequences] * 8)
        return X, y
    
    def _build_model(self) -> keras.Model:
        """Build model."""
        inputs = layers.Input(shape=(8,), dtype="int32")
        x = TokenAndPositionEmbedding(8, 20, 32)(inputs)
        x = TransformerBlock(32, 4, 128)(x)
        outputs = layers.Dense(20)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model
    
    def demonstrate_accumulation(self) -> None:
        """Show gradient accumulation concept."""
        print("=" * 70)
        print("🔄 GRADIENT ACCUMULATION")
        print("=" * 70)
        print()
        
        print("THE PROBLEM:")
        print("  You want batch_size=32 for stable training")
        print("  But GPU only fits batch_size=8")
        print("  Solution: Accumulate gradients over 4 steps")
        print()
        
        print("THE CONCEPT:")
        print("  ┌───────────────────────────────────────────────┐")
        print("  │  Normal training (batch_size=32):              │")
        print("  │    Compute gradients on 32 samples           │")
        print("  │    Update weights                            │")
        print("  └───────────────────────────────────────────────┘")
        print()
        print("  ┌───────────────────────────────────────────────┐")
        print("  │  Gradient accumulation (batch_size=8, steps=4):│")
        print("  │    Step 1: Compute gradients on samples 0-7    │")
        print("  │           Accumulate (add to running sum)      │")
        print("  │    Step 2: Compute gradients on samples 8-15   │")
        print("  │           Accumulate                           │")
        print("  │    Step 3: Compute gradients on samples 16-23│")
        print("  │           Accumulate                           │")
        print("  │    Step 4: Compute gradients on samples 24-31  │")
        print("  │           Accumulate                           │")
        print("  │    ─────────────────────────────────────────   │")
        print("  │    Update weights with accumulated gradients   │")
        print("  └───────────────────────────────────────────────┘")
        print()
        
        print("MATHEMATICAL EQUIVALENCE:")
        print("  Effective batch size = batch_per_step × accumulation_steps")
        print("  Memory usage = batch_per_step (not effective batch!)")
        print()
        
        # Simulation
        print("SIMULATION: Training with different strategies")
        print()
        
        strategies = [
            ('Small batch (4)', 4, 1),
            ('Medium batch (8)', 8, 1),
            ('Accumulation (8×4)', 8, 4),
            ('Full batch (32)', 32, 1),
        ]
        
        results = []
        for name, batch_size, accum_steps in strategies:
            model = self._build_model()
            
            # Simulate training (just a few steps)
            effective_batch = batch_size * accum_steps
            
            # Estimate memory (simplified)
            memory_per_sample = 0.5  # MB (rough estimate)
            memory_used = batch_size * memory_per_sample
            
            print(f"  {name}:")
            print(f"    Batch per step: {batch_size}")
            print(f"    Accumulation steps: {accum_steps}")
            print(f"    Effective batch: {effective_batch}")
            print(f"    Memory: ~{memory_used:.1f} MB")
            print()
            
            results.append({
                'name': name,
                'effective_batch': effective_batch,
                'memory': memory_used
            })
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [r['name'] for r in results]
        memory = [r['memory'] for r in results]
        effective = [r['effective_batch'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, memory, width, label='Memory Used (MB)', color='blue', alpha=0.7)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, effective, width, label='Effective Batch Size', color='green', alpha=0.7)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Memory (MB)', color='blue')
        ax2.set_ylabel('Effective Batch Size', color='green')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Gradient Accumulation: Same Effective Batch, Less Memory')
        plt.tight_layout()
        plt.savefig('gradient_accumulation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'gradient_accumulation.png'")
        print()
        
        print("KEY INSIGHT:")
        print("  Accumulation (8×4) uses 4× less memory than full batch (32)")
        print("  But has same training dynamics!")
        print("  Trade-off: Takes 4× longer per update")
        print()


class MixedPrecisionDemo:
    """Demonstrates mixed precision training concepts."""
    
    def explain_mixed_precision(self) -> None:
        """Explain FP16/BF16 training."""
        print("=" * 70)
        print("🔢 MIXED PRECISION TRAINING (FP16/BF16)")
        print("=" * 70)
        print()
        
        print("THE PROBLEM:")
        print("  FP32 (32-bit floats) use 4 bytes per number")
        print("  Large models have billions of parameters")
        print("  Memory is the bottleneck!")
        print()
        
        print("THE SOLUTION: Mixed Precision")
        print("  Use FP16 (16-bit) or BF16 (16-bit brain float) where possible")
        print()
        
        print("FORMAT COMPARISON:")
        print("  ┌──────────┬────────────┬────────────┬──────────────┐")
        print("  │ Format   │ Exponent   │ Mantissa   │ Precision    │")
        print("  ├──────────┼────────────┼────────────┼──────────────┤")
        print("  │ FP32     │ 8 bits     │ 23 bits    │ ~7 decimals  │")
        print("  │ FP16     │ 5 bits     │ 10 bits    │ ~3 decimals  │")
        print("  │ BF16     │ 8 bits     │ 7 bits     │ ~2 decimals  │")
        print("  └──────────┴────────────┴────────────┴──────────────┘")
        print()
        
        print("TRADE-OFFS:")
        print("  FP16:")
        print("    ✅ 2× memory savings")
        print("    ✅ 2× speedup on tensor cores")
        print("    ❌ Limited range (can overflow/underflow)")
        print("    ❌ Need loss scaling")
        print()
        print("  BF16:")
        print("    ✅ 2× memory savings")
        print("    ✅ Same range as FP32 (no overflow)")
        print("    ❌ Less precision than FP16")
        print("    ❌ Not supported on all hardware")
        print()
        
        print("MIXED PRECISION STRATEGY:")
        print("  Forward pass:     FP16/BF16 (fast, memory-efficient)")
        print("  Loss calculation: FP32 (precise)")
        print("  Backward pass:    FP16/BF16 (fast)")
        print("  Weight update:    FP32 (precise)")
        print("  Master weights:   FP32 (always precise copy)")
        print()
        
        print("MEMORY SAVINGS:")
        print("  Model:     50% reduction (FP16 vs FP32)")
        print("  Activations: 50% reduction")
        print("  Gradients:   50% reduction")
        print("  Total:     ~50% training memory saved!")
        print()
        
        print("SPEEDUP:")
        print("  Modern GPUs (V100, A100, H100) have tensor cores")
        print("  Tensor cores do FP16 matrix multiply 8× faster!")
        print("  Real-world: 2-3× training speedup")
        print()
        
        print("IN YOUR CODEBASE:")
        print("  • Keras: keras.mixed_precision.set_global_policy('mixed_float16')")
        print("  • PyTorch: torch.cuda.amp.autocast()")
        print("  • Automatic loss scaling prevents underflow")
        print()
        
        # Visualization
        formats = ['FP32', 'FP16', 'BF16']
        memory = [100, 50, 50]  # Percentage
        speed = [100, 250, 250]  # Percentage (2.5x with tensor cores)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Memory comparison
        ax1.bar(formats, memory, color=['blue', 'green', 'orange'], alpha=0.7)
        ax1.set_ylabel('Memory Usage (%)')
        ax1.set_title('Memory Usage by Format')
        ax1.set_ylim(0, 120)
        for i, v in enumerate(memory):
            ax1.text(i, v + 3, f'{v}%', ha='center', fontweight='bold')
        
        # Speed comparison
        ax2.bar(formats, speed, color=['blue', 'green', 'orange'], alpha=0.7)
        ax2.set_ylabel('Relative Speed (%)')
        ax2.set_title('Training Speed by Format\n(with tensor cores)')
        ax2.set_ylim(0, 300)
        ax2.axhline(y=100, color='red', linestyle='--', label='FP32 baseline')
        for i, v in enumerate(speed):
            ax2.text(i, v + 5, f'{v}%', ha='center', fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('mixed_precision.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'mixed_precision.png'")
        print()


class BatchSizeEffects:
    """Demonstrate batch size effects on training."""
    
    def demonstrate_batch_size(self) -> None:
        """Show how batch size affects training."""
        print("=" * 70)
        print("📦 BATCH SIZE EFFECTS")
        print("=" * 70)
        print()
        
        print("TRADE-OFFS:")
        print()
        print("  Small batch (e.g., 4):")
        print("    ✅ Lower memory")
        print("    ✅ More updates per epoch (better exploration)")
        print("    ❌ Noisy gradients (high variance)")
        print("    ❌ Slower training (less parallelism)")
        print()
        print("  Large batch (e.g., 1024):")
        print("    ✅ Stable gradients (low variance)")
        print("    ✅ Fast training (good parallelism)")
        print("    ❌ High memory")
        print("    ❌ Can get stuck in sharp minima")
        print("    ❌ Need learning rate warmup")
        print()
        print("  Sweet spot: 32-256 (depends on model/dataset)")
        print()
        
        # Simulate training curves
        np.random.seed(42)
        epochs = np.arange(50)
        
        # Small batch: noisy but good generalization
        small_batch = 1 - np.exp(-epochs/15) + np.random.randn(50) * 0.05
        small_batch = np.clip(small_batch, 0, 1)
        
        # Medium batch: balanced
        medium_batch = 1 - np.exp(-epochs/12) + np.random.randn(50) * 0.03
        medium_batch = np.clip(medium_batch, 0, 1)
        
        # Large batch: fast initial progress, may plateau
        large_batch = 1 - np.exp(-epochs/8) + np.random.randn(50) * 0.02
        large_batch = np.clip(large_batch, 0, 1)
        large_batch[30:] = large_batch[30] + np.random.randn(20) * 0.01  # Plateau
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, small_batch, label='Batch=4 (small)', linewidth=2, alpha=0.8)
        ax.plot(epochs, medium_batch, label='Batch=32 (medium)', linewidth=2, alpha=0.8)
        ax.plot(epochs, large_batch, label='Batch=256 (large)', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Dynamics vs Batch Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('batch_size_effects.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'batch_size_effects.png'")
        print()
        
        print("MODERN BEST PRACTICES:")
        print("  1. Start with batch_size that fits in memory")
        print("  2. Use gradient accumulation if need larger effective batch")
        print("  3. Scale learning rate with batch size (linear scaling rule)")
        print("  4. Use warmup when increasing batch size")
        print("  5. Try different sizes and measure wall-clock time to convergence")
        print()


def main():
    """Main execution for Week 5."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 22 + "WEEK 5: SCALE" + " " * 33 + "║")
    print("║" + " " * 12 + "Memory, Efficiency & Large Model Training" + " " * 13 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Memory profiling
    profiler = MemoryProfiler()
    profiler.calculate_model_memory(
        vocab_size=1000,
        max_length=128,
        embedding_dim=256,
        num_blocks=4,
        num_heads=8
    )
    
    profiler.compare_model_sizes()
    
    # Gradient accumulation
    accum_demo = GradientAccumulationDemo()
    accum_demo.demonstrate_accumulation()
    
    # Mixed precision
    mp_demo = MixedPrecisionDemo()
    mp_demo.explain_mixed_precision()
    
    # Batch size effects
    batch_demo = BatchSizeEffects()
    batch_demo.demonstrate_batch_size()
    
    print("=" * 70)
    print("🎯 WEEK 5 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. MEMORY BREAKDOWN:")
    print("   • Model weights: ~25% of training memory")
    print("   • Activations: ~25% (forward pass)")
    print("   • Gradients: ~25% (backward pass)")
    print("   • Optimizer states: ~25% (Adam: 2× model)")
    print()
    print("2. GRADIENT ACCUMULATION:")
    print("   • Simulate large batches with limited memory")
    print("   • Effective batch = per_step_batch × num_steps")
    print("   • Trade-off: time vs memory")
    print()
    print("3. MIXED PRECISION:")
    print("   • FP16/BF16: 50% memory, 2-3× speedup")
    print("   • Essential for large models")
    print("   • BF16 preferred (better range than FP16)")
    print()
    print("4. BATCH SIZE:")
    print("   • Small: noisy gradients, better generalization")
    print("   • Large: stable gradients, may need warmup")
    print("   • Sweet spot: 32-256 (problem-dependent)")
    print()
    print("5. SCALE MATTERS:")
    print("   • GPT-3: 175B params = 700GB just for weights!")
    print("   • Training requires 4-8× model size in memory")
    print("   • Distributed training across many GPUs essential")
    print()
    print("=" * 70)
    print()
    print("Next: Week 6 - Distributed Training & Model Parallelism")
    print("      Or explore your training setup with these optimizations!")
    print()


if __name__ == "__main__":
    main()
