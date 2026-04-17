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

"""Week 4: Optimizers & Learning Rate - The Art of Training.

This script compares different optimizers and learning rate strategies
to build deep intuition for what makes training work well vs poorly.

Key questions answered:
- Why Adam instead of SGD?
- What learning rate should I use?
- How do learning rate schedules help?
- What is gradient clipping and why does it matter?
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax
import jax.numpy as jnp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
import sys

sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')

from transformers.layers import (
    TokenAndPositionEmbedding,
    TransformerBlock,
)


class OptimizerComparison:
    """Compares different optimizers on the same task."""
    
    def __init__(self, vocab_size: int = 20, embedding_dim: int = 32,
                 max_length: int = 8):
        """Initialize comparison experiment."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.X, self.y = self._get_dataset()
        
    def _get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create training dataset."""
        sequences = [
            ([1, 2, 3, 4, 1, 2, 3, 4], [2, 3, 4, 1, 2, 3, 4, 1]),
            ([5, 6, 5, 6, 5, 6, 5, 6], [6, 5, 6, 5, 6, 5, 6, 5]),
            ([7, 8, 9, 7, 8, 9, 7, 8], [8, 9, 7, 8, 9, 7, 8, 9]),
            ([10, 11, 10, 11, 10, 11, 10, 11], [11, 10, 11, 10, 11, 10, 11, 10]),
            ([12, 13, 14, 15, 12, 13, 14, 15], [13, 14, 15, 12, 13, 14, 15, 12]),
        ]
        X = np.array([s[0] for s in sequences])
        y = np.array([s[1] for s in sequences])
        return X, y
    
    def _build_model(self) -> keras.Model:
        """Build fresh model instance."""
        inputs = layers.Input(shape=(self.max_length,), dtype="int32")
        
        x = TokenAndPositionEmbedding(
            max_length=self.max_length,
            vocabulary_size=self.vocab_size,
            embedding_dim=self.embedding_dim
        )(inputs)
        
        for _ in range(2):
            x = TransformerBlock(
                embedding_dim=self.embedding_dim,
                num_heads=4,
                mlp_dim=self.embedding_dim * 4,
                dropout_rate=0.0
            )(x)
        
        outputs = layers.Dense(self.vocab_size)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_with_optimizer(self, optimizer: keras.optimizers.Optimizer,
                           name: str, epochs: int = 50) -> Dict:
        """Train model with specific optimizer."""
        model = self._build_model()
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=5,
            verbose=0
        )
        
        return {
            'name': name,
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'final_loss': history.history['loss'][-1],
            'final_acc': history.history['accuracy'][-1]
        }
    
    def compare_optimizers(self) -> None:
        """Compare SGD vs Momentum vs Adam vs AdamW."""
        print("=" * 70)
        print("⚔️  OPTIMIZER BATTLE ROYALE")
        print("=" * 70)
        print()
        print("Training identical models with different optimizers...")
        print("Task: Learn 5 simple patterns")
        print()
        
        optimizers = [
            (keras.optimizers.SGD(learning_rate=0.01), "SGD (lr=0.01)"),
            (keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), "SGD + Momentum"),
            (keras.optimizers.Adam(learning_rate=0.001), "Adam (lr=0.001)"),
            (keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01), "AdamW"),
        ]
        
        results = []
        for opt, name in optimizers:
            print(f"  Training with {name}...", end=" ")
            result = self.train_with_optimizer(opt, name, epochs=50)
            results.append(result)
            print(f"✅ Final acc: {result['final_acc']:.1%}")
        
        print()
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        ax1 = axes[0]
        for result in results:
            ax1.plot(result['loss'], label=result['name'], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves: Optimizer Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Accuracy curves
        ax2 = axes[1]
        for result in results:
            ax2.plot(result['accuracy'], label=result['name'], linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves: Optimizer Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Comparison saved to 'optimizer_comparison.png'")
        print()
        
        # Summary table
        print("RESULTS SUMMARY:")
        print("-" * 50)
        print(f"{'Optimizer':<20} {'Final Loss':<12} {'Final Acc':<12}")
        print("-" * 50)
        for result in results:
            print(f"{result['name']:<20} {result['final_loss']:<12.4f} {result['final_acc']:<12.1%}")
        print("-" * 50)
        print()


class LearningRateDemo:
    """Demonstrates learning rate effects."""
    
    def __init__(self):
        """Initialize demo."""
        self.X = np.array([[1, 2, 3, 4]], dtype=np.float32)
        self.y = np.array([[2, 4, 6, 8]], dtype=np.float32)  # y = 2x
        
    def _build_simple_model(self) -> keras.Model:
        """Build a simple linear model."""
        model = keras.Sequential([
            layers.Dense(1, use_bias=False, input_shape=(1,))
        ])
        return model
    
    def demonstrate_lr_effects(self) -> None:
        """Show what happens with different learning rates."""
        print("=" * 70)
        print("📊 LEARNING RATE: THE GOLDILOCKS PROBLEM")
        print("=" * 70)
        print()
        print("Task: Learn y = 2x (find weight = 2)")
        print()
        
        learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
        results = []
        
        for lr in learning_rates:
            model = self._build_simple_model()
            model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss='mse')
            
            # Reshape data
            X = self.X.reshape(-1, 1)
            y = self.y.reshape(-1, 1)
            
            # Get initial weight
            initial_weight = model.layers[0].get_weights()[0][0, 0]
            
            # Train
            history = model.fit(X, y, epochs=20, verbose=0)
            
            # Get final weight
            final_weight = model.layers[0].get_weights()[0][0, 0]
            
            results.append({
                'lr': lr,
                'initial': initial_weight,
                'final': final_weight,
                'losses': history.history['loss']
            })
            
            status = "✅" if abs(final_weight - 2.0) < 0.1 else "⚠️"
            print(f"  LR = {lr:5.3f}: Initial={initial_weight:+.2f} → Final={final_weight:+.2f} {status}")
        
        print()
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for result in results:
            label = f"LR = {result['lr']}"
            if result['lr'] == 0.001:
                label += " (too small - slow)"
            elif result['lr'] == 0.5:
                label += " (too big - unstable)"
            elif result['lr'] == 0.1:
                label += " (just right)"
            
            ax.plot(result['losses'], label=label, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Learning Rate Effects: Finding the Sweet Spot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('learning_rate_effects.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'learning_rate_effects.png'")
        print()


class LRSchedulerDemo:
    """Demonstrates learning rate schedules."""
    
    def visualize_schedules(self) -> None:
        """Visualize different LR schedules."""
        print("=" * 70)
        print("📅 LEARNING RATE SCHEDULES")
        print("=" * 70)
        print()
        print("Why change learning rate during training?")
        print("  • Start: Large steps to make progress quickly")
        print("  • End: Small steps to fine-tune precisely")
        print()
        
        epochs = np.arange(100)
        initial_lr = 0.1
        
        schedules = {
            'Constant': np.full_like(epochs, initial_lr, dtype=float),
            'Step Decay': initial_lr * (0.5 ** (epochs // 30)),
            'Exponential': initial_lr * np.exp(-0.03 * epochs),
            'Cosine': initial_lr * (0.5 * (1 + np.cos(np.pi * epochs / 100))),
            'Warmup + Cosine': np.where(
                epochs < 10,
                initial_lr * epochs / 10,  # Linear warmup
                initial_lr * (0.5 * (1 + np.cos(np.pi * (epochs - 10) / 90)))
            )
        }
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for (name, schedule), color in zip(schedules.items(), colors):
            ax.plot(epochs, schedule, label=name, linewidth=2, color=color)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Common Learning Rate Schedules')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lr_schedules.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Schedules saved to 'lr_schedules.png'")
        print()
        
        print("Schedule descriptions:")
        print("  Constant:     Same LR throughout (simple, often suboptimal)")
        print("  Step Decay:   Reduce by half every 30 epochs")
        print("  Exponential:  Smooth continuous decay")
        print("  Cosine:       Smooth decay following cosine curve")
        print("  Warmup+Cosine: Small steps at start, then cosine decay")
        print("                 → Most popular for large models (GPT, Llama)")
        print()


class GradientClippingDemo:
    """Demonstrates gradient clipping."""
    
    def demonstrate_exploding_gradients(self) -> None:
        """Show why gradient clipping is needed."""
        print("=" * 70)
        print("✂️  GRADIENT CLIPPING: Preventing Explosions")
        print("=" * 70)
        print()
        
        print("The Problem: Exploding Gradients")
        print("  In deep networks, gradients can become HUGE")
        print("  This causes unstable training or NaN losses")
        print()
        
        print("The Solution: Gradient Clipping")
        print("  If gradient norm > threshold:")
        print("    gradient = gradient × (threshold / norm)")
        print("  This keeps gradients bounded")
        print()
        
        # Simulate gradient norms during training
        np.random.seed(42)
        epochs = 50
        
        # Simulated gradient norms (with occasional spikes)
        unclipped = []
        clipped = []
        threshold = 5.0
        
        for epoch in range(epochs):
            # Normal gradients + occasional spikes
            base_norm = 2.0 + np.random.randn() * 0.5
            if epoch in [15, 32]:  # Explosion events
                base_norm *= 10
            
            unclipped.append(base_norm)
            clipped.append(min(base_norm, threshold))
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(unclipped, label='Unclipped (explosions!)', 
                color='red', linewidth=2, alpha=0.7)
        ax.plot(clipped, label=f'Clipped (max={threshold})', 
                color='green', linewidth=2)
        ax.axhline(y=threshold, color='orange', linestyle='--', 
                  label=f'Clip threshold = {threshold}')
        
        # Mark explosion points
        for epoch in [15, 32]:
            ax.annotate('EXPLOSION!', xy=(epoch, unclipped[epoch]),
                       xytext=(epoch+5, unclipped[epoch]+5),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       color='red', fontweight='bold')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Clipping Prevents Explosions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gradient_clipping.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'gradient_clipping.png'")
        print()
        
        print("In your codebase:")
        print("  • Keras: optimizer with clipnorm=1.0 or clipvalue=1.0")
        print("  • PyTorch: torch.nn.utils.clip_grad_norm_()")
        print("  • Essential for training deep transformers!")
        print()


def explain_optimizers() -> None:
    """Explain different optimizers conceptually."""
    print("=" * 70)
    print("📚 OPTIMIZER DEEP DIVE")
    print("=" * 70)
    print()
    
    print("1. SGD (Stochastic Gradient Descent)")
    print("   Update: weight = weight - lr × gradient")
    print("   Analogy: Walking downhill by feeling the slope")
    print("   Pros: Simple, well-understood")
    print("   Cons: Slow, gets stuck in valleys, zigzags")
    print()
    
    print("2. SGD + Momentum")
    print("   Update: velocity = momentum × velocity + gradient")
    print("           weight = weight - lr × velocity")
    print("   Analogy: Rolling ball that builds up speed")
    print("   Pros: Faster, escapes shallow valleys")
    print("   Cons: Can overshoot")
    print()
    
    print("3. Adam (Adaptive Moment Estimation)")
    print("   Key ideas:")
    print("     • Momentum: remember past gradients (like SGD+Momentum)")
    print("     • Adaptive LR: different learning rate per parameter")
    print("   Analogy: Each parameter has its own speedometer")
    print("   Pros: Fast convergence, works well out-of-the-box")
    print("   Cons: Can overfit, needs weight decay variant (AdamW)")
    print()
    
    print("4. AdamW (Adam + Weight Decay)")
    print("   Difference: Decouples weight decay from gradient update")
    print("   Effect: Better regularization, less overfitting")
    print("   Status: Default choice for most LLMs (GPT, Llama, etc.)")
    print()
    
    print("RECOMMENDATION:")
    print("  → Start with: Adam (lr=1e-3 or 1e-4)")
    print("  → For final runs: AdamW with weight decay")
    print("  → For huge batches: LAMB or LARS (specialized variants)")
    print()


def main():
    """Main execution for Week 4."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "WEEK 4: OPTIMIZATION" + " " * 27 + "║")
    print("║" + " " * 15 + "The Art of Training Well" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Conceptual explanations
    explain_optimizers()
    
    # Optimizer comparison
    print("Running optimizer comparison experiment...")
    print("(This may take 1-2 minutes)")
    print()
    
    comparison = OptimizerComparison()
    comparison.compare_optimizers()
    
    # Learning rate effects
    lr_demo = LearningRateDemo()
    lr_demo.demonstrate_lr_effects()
    
    # LR schedules
    scheduler = LRSchedulerDemo()
    scheduler.visualize_schedules()
    
    # Gradient clipping
    clipping = GradientClippingDemo()
    clipping.demonstrate_exploding_gradients()
    
    print("=" * 70)
    print("🎯 WEEK 4 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. OPTIMIZER CHOICE:")
    print("   • Adam: Good default, adaptive per-parameter")
    print("   • AdamW: Better regularization, preferred for LLMs")
    print("   • SGD+Momentum: Classic, sometimes more stable")
    print()
    print("2. LEARNING RATE:")
    print("   • Too small: Training takes forever")
    print("   • Too large: Unstable, might diverge")
    print("   • Just right: Steady progress to convergence")
    print("   • Typical range: 1e-5 to 1e-3 for transformers")
    print()
    print("3. LEARNING RATE SCHEDULES:")
    print("   • Start with warmup (small steps)")
    print("   • Decay over time (cosine or linear)")
    print("   • Modern standard: warmup + cosine decay")
    print()
    print("4. GRADIENT CLIPPING:")
    print("   • Prevents exploding gradients")
    print("   • Essential for training deep transformers")
    print("   • Common threshold: clipnorm=1.0")
    print()
    print("5. YOUR CODEBASE:")
    print("   • training/model.py: Supports AdamW (line 143-147)")
    print("   • Can add gradient clipping to optimizer config")
    print("   • Add learning rate scheduler for longer training")
    print()
    print("=" * 70)
    print()
    print("Next: Week 5 - Scale & Compute Efficiency")
    print("      (Mixed precision, gradient accumulation, memory profiling)")
    print()


if __name__ == "__main__":
    main()
