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

"""Week 3: Training Loop & The Forward Pass.

This script demonstrates how LLMs learn through:
1. Forward pass: input → prediction
2. Loss calculation: how wrong are we?
3. Backward pass: calculate gradients
4. Update: adjust weights to reduce loss

We'll train a tiny model to overfit to a small dataset,
which is the best way to understand training dynamics.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax
import jax.numpy as jnp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys

sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')

from transformers.layers import (
    TokenAndPositionEmbedding,
    TransformerBlock,
)


class SimpleTrainingDemo:
    """Demonstrates training on a tiny dataset to build intuition."""
    
    def __init__(self, vocab_size: int = 20, embedding_dim: int = 32,
                 max_length: int = 8, num_blocks: int = 2):
        """Initialize training demo.
        
        Args:
            vocab_size: Small vocabulary for demo
            embedding_dim: Embedding dimension
            max_length: Max sequence length
            num_blocks: Number of transformer blocks
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_blocks = num_blocks
        
        # Build model
        self.model = self._build_model()
        
        # Training history
        self.history = {'loss': [], 'accuracy': []}
        
    def _build_model(self) -> keras.Model:
        """Build a tiny transformer model."""
        inputs = layers.Input(shape=(self.max_length,), dtype="int32")
        
        # Embedding
        x = TokenAndPositionEmbedding(
            max_length=self.max_length,
            vocabulary_size=self.vocab_size,
            embedding_dim=self.embedding_dim
        )(inputs)
        
        # Transformer blocks
        for _ in range(self.num_blocks):
            x = TransformerBlock(
                embedding_dim=self.embedding_dim,
                num_heads=4,
                mlp_dim=self.embedding_dim * 4,
                dropout_rate=0.0
            )(x)
        
        # Output
        outputs = layers.Dense(self.vocab_size)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        return model
    
    def get_toy_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a tiny toy dataset for overfitting.
        
        Returns:
            X: Input sequences [num_samples, max_length]
            y: Target tokens [num_samples, max_length]
        """
        # Dataset: sequences and their next-token targets
        # We'll use simple patterns the model can learn
        
        sequences = [
            # Pattern: 1, 2, 3, 4 repeats
            [1, 2, 3, 4, 1, 2, 3, 4],  # Input
            [2, 3, 4, 1, 2, 3, 4, 1],  # Target (shifted by 1)
            
            # Pattern: 5, 6, 5, 6 alternates
            [5, 6, 5, 6, 5, 6, 5, 6],
            [6, 5, 6, 5, 6, 5, 6, 5],
            
            # Pattern: 7, 8, 9 counts
            [7, 8, 9, 7, 8, 9, 7, 8],
            [8, 9, 7, 8, 9, 7, 8, 9],
            
            # Pattern: 10, 11 repeats
            [10, 11, 10, 11, 10, 11, 10, 11],
            [11, 10, 11, 10, 11, 10, 11, 10],
            
            # Pattern: 12, 13, 14, 15
            [12, 13, 14, 15, 12, 13, 14, 15],
            [13, 14, 15, 12, 13, 14, 15, 12],
        ]
        
        X = np.array([seq for i, seq in enumerate(sequences) if i % 2 == 0])
        y = np.array([seq for i, seq in enumerate(sequences) if i % 2 == 1])
        
        return X, y
    
    def train_step_visual(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform one training step with detailed logging.
        
        Args:
            X: Input batch [batch_size, seq_len]
            y: Target batch [batch_size, seq_len]
            
        Returns:
            Metrics dict with loss and accuracy
        """
        # Forward pass + backward pass + weight update
        # All handled by model.train_on_batch
        loss, accuracy = self.model.train_on_batch(X, y)
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def train(self, epochs: int = 100, verbose: bool = True) -> None:
        """Train the model and visualize progress.
        
        Args:
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        X, y = self.get_toy_dataset()
        
        print("=" * 70)
        print("🎯 TRAINING: OVERFITTING TO TOY DATASET")
        print("=" * 70)
        print()
        
        print("Dataset:")
        print(f"  Number of sequences: {len(X)}")
        print(f"  Sequence length: {self.max_length}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print()
        
        print("Model architecture:")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Transformer blocks: {self.num_blocks}")
        print(f"  Parameters: {self.model.count_params():,}")
        print()
        
        # Show sample data
        print("Sample training data:")
        for i in range(min(3, len(X))):
            print(f"  Input:  {X[i].tolist()}")
            print(f"  Target: {y[i].tolist()}")
            print()
        
        print("Training...")
        print()
        
        # Training loop
        for epoch in range(epochs):
            metrics = self.train_step_visual(X, y)
            
            self.history['loss'].append(metrics['loss'])
            self.history['accuracy'].append(metrics['accuracy'])
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}: loss={metrics['loss']:.4f}, "
                      f"accuracy={metrics['accuracy']:.2%}")
        
        print()
        print("✅ Training complete!")
        print()
    
    def visualize_training(self) -> None:
        """Visualize training progress."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1 = axes[0]
        ax1.plot(self.history['loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Accuracy plot
        ax2 = axes[1]
        ax2.plot(self.history['accuracy'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy Over Time')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Training visualization saved to 'training_progress.png'")
        print()
    
    def test_predictions(self) -> None:
        """Test model predictions on training data."""
        print("=" * 70)
        print("🔮 TESTING PREDICTIONS")
        print("=" * 70)
        print()
        
        X, y = self.get_toy_dataset()
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        
        print("Predictions vs Targets:")
        for i in range(len(X)):
            pred_ids = np.argmax(predictions[i], axis=-1)
            target_ids = y[i]
            
            match = np.all(pred_ids == target_ids)
            symbol = "✅" if match else "❌"
            
            print(f"  Sample {i+1}:")
            print(f"    Input:     {X[i].tolist()}")
            print(f"    Predicted: {pred_ids.tolist()}")
            print(f"    Target:    {target_ids.tolist()}")
            print(f"    Match:     {symbol}")
            print()
        
        # Calculate overall accuracy
        all_correct = 0
        total = 0
        for i in range(len(X)):
            pred_ids = np.argmax(predictions[i], axis=-1)
            target_ids = y[i]
            all_correct += np.sum(pred_ids == target_ids)
            total += len(target_ids)
        
        accuracy = all_correct / total
        print(f"Overall token-level accuracy: {accuracy:.2%}")
        print()


def explain_training_concept() -> None:
    """Explain how training works."""
    print("=" * 70)
    print("📚 HOW LLMs LEARN: THE TRAINING LOOP")
    print("=" * 70)
    print()
    
    print("THE FOUR STEPS OF TRAINING:")
    print()
    print("1️⃣  FORWARD PASS")
    print("   Input: [1, 2, 3, 4, 1, 2, 3, 4]")
    print("     ↓")
    print("   Model: Embedding → Transformer → Dense")
    print("     ↓")
    print("   Output: logits for each position")
    print("     [logits_pos0, logits_pos1, ..., logits_pos7]")
    print("   Each logit is a score for every possible next token")
    print()
    
    print("2️⃣  LOSS CALCULATION")
    print("   Target: [2, 3, 4, 1, 2, 3, 4, 1]")
    print("   Prediction: [2.5, 2.1, 1.8, ...] (softmax probabilities)")
    print("   Loss: -log(probability of correct token)")
    print("   → Measures 'how surprised' the model is by the correct answer")
    print("   → Lower loss = better predictions")
    print()
    
    print("3️⃣  BACKWARD PASS (BACKPROPAGATION)")
    print("   Question: Which weights contributed to the error?")
    print("   Answer: Calculate gradient (derivative) of loss w.r.t. each weight")
    print("   Gradient tells us: 'increase this weight → loss goes up/down'")
    print()
    
    print("4️⃣  WEIGHT UPDATE (GRADIENT DESCENT)")
    print("   weight_new = weight_old - learning_rate × gradient")
    print("   If gradient is positive: decrease weight (reduces loss)")
    print("   If gradient is negative: increase weight (reduces loss)")
    print("   Learning rate: step size (typically 1e-4 to 1e-3)")
    print()
    
    print("KEY INSIGHT:")
    print("  Training is just minimizing loss through iterative adjustment.")
    print("  Each step makes the model slightly better at predicting.")
    print("  After thousands of steps → powerful model!")
    print()
    
    print("WHY OVERFIT TO SMALL DATA FIRST?")
    print("  • Fast feedback loop (train in seconds, not hours)")
    print("  • Understand if your model can learn at all")
    print("  • Debug issues before scaling up")
    print("  • Build intuition about training dynamics")
    print()


def explain_loss_functions() -> None:
    """Explain different loss functions."""
    print("=" * 70)
    print("🎯 LOSS FUNCTIONS FOR LANGUAGE MODELS")
    print("=" * 70)
    print()
    
    print("1. CROSS-ENTROPY LOSS (what we use)")
    print("   Formula: -Σ(target × log(prediction))")
    print("   Intuition: Measures surprise at correct answer")
    print("   • If model predicts correct token with 100%: loss = 0")
    print("   • If model predicts correct token with 50%: loss = 0.69")
    print("   • If model predicts correct token with 1%: loss = 4.6")
    print("   → Penalizes confident wrong predictions heavily")
    print()
    
    print("2. PERPLEXITY (derived from cross-entropy)")
    print("   Formula: exp(cross_entropy)")
    print("   Intuition: 'Effective vocabulary size'")
    print("   • Perplexity = 100: model as confused as uniform over 100 tokens")
    print("   • Perplexity = 10: model is pretty certain")
    print("   • Lower is better")
    print()
    
    print("3. MASKED LOSS (for padding)")
    print("   Problem: Padding tokens shouldn't contribute to loss")
    print("   Solution: Multiply loss by mask (0 for padding, 1 for real)")
    print("   Your codebase: CustomMaskPadLoss in training/losses.py")
    print()


def main():
    """Main execution for Week 3."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "WEEK 3: TRAINING LOOP" + " " * 27 + "║")
    print("║" + " " * 12 + "Learning Through Gradient Descent" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Conceptual explanation
    explain_training_concept()
    explain_loss_functions()
    
    # Practical demonstration
    print("=" * 70)
    print("🧪 PRACTICAL DEMONSTRATION: TRAINING A TINY MODEL")
    print("=" * 70)
    print()
    
    # Create and train
    demo = SimpleTrainingDemo(
        vocab_size=20,
        embedding_dim=32,
        max_length=8,
        num_blocks=2
    )
    
    demo.train(epochs=100, verbose=True)
    demo.visualize_training()
    demo.test_predictions()
    
    print("=" * 70)
    print("🎯 WEEK 3 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. THE TRAINING LOOP:")
    print("   Forward → Loss → Backward → Update")
    print("   Repeat thousands of times")
    print()
    print("2. GRADIENT DESCENT INTUITION:")
    print("   Imagine loss as a landscape with hills and valleys")
    print("   Training = walking downhill to find the lowest point")
    print("   Gradient = direction of steepest descent")
    print()
    print("3. CROSS-ENTROPY LOSS:")
    print("   Measures surprise at the correct answer")
    print("   Minimizing it = maximizing prediction confidence")
    print()
    print("4. OVERFITTING SMALL DATA:")
    print("   Essential debugging technique")
    print("   If model can't memorize 5 examples, something's wrong!")
    print()
    print("5. YOUR CODEBASE:")
    print("   • training/model.py: create_model() for model building")
    print("   • training/losses.py: CustomMaskPadLoss for masked loss")
    print("   • training/callbacks.py: CustomAccuracyPrinter for metrics")
    print()
    print("=" * 70)
    print()
    print("Next: Week 4 - Optimization (Adam, learning rates, schedules)")
    print("      Or try modifying the toy dataset and retraining!")
    print()


if __name__ == "__main__":
    main()
