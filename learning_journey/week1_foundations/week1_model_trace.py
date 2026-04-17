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

"""Week 1: Model Trace - See Real Tensors Flow Through Your Code.

This script creates a TINY transformer model and traces tensor shapes
through each layer. It connects the conceptual understanding from
week1_attention_visualization.py to the actual code in your repo.

Run this after you've studied the conceptual visualization.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax
import jax.numpy as jnp
import keras
from keras import layers

# Import your actual layers from the codebase
import sys
sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')

from transformers.layers import (
    TokenAndPositionEmbedding,
    TransformerBlock,
    MultiHeadSelfAttention,
    FeedForwardNetwork,
)

# Disable GPU for clearer logging (optional)
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def trace_tensor(name: str, tensor: jax.Array, indent: int = 0) -> None:
    """Pretty-prints tensor information."""
    prefix = "  " * indent
    shape_str = " × ".join(str(x) for x in tensor.shape)
    dtype_str = str(tensor.dtype).replace('jax.numpy.', '')
    
    # Calculate memory usage
    num_elements = tensor.size
    bytes_per_element = tensor.dtype.itemsize
    memory_kb = (num_elements * bytes_per_element) / 1024
    
    print(f"{prefix}📦 {name}")
    print(f"{prefix}   Shape: [{shape_str}]")
    print(f"{prefix}   Dtype: {dtype_str}")
    print(f"{prefix}   Elements: {num_elements:,}")
    print(f"{prefix}   Memory: ~{memory_kb:.1f} KB")
    print()


def demo_embeddings() -> None:
    """Demonstrates TokenAndPositionEmbedding with shape tracing."""
    print("=" * 70)
    print("🔤 STEP 1: TOKEN & POSITION EMBEDDINGS")
    print("=" * 70)
    print()
    
    # Tiny configuration for clarity
    vocab_size = 20  # Only 20 possible tokens
    max_length = 8   # Max 8 tokens per sequence
    embedding_dim = 16  # Each token becomes 16-dimensional vector
    
    print("Configuration:")
    print(f"  • Vocabulary size: {vocab_size} (tiny for demo)")
    print(f"  • Max sequence length: {max_length}")
    print(f"  • Embedding dimension: {embedding_dim}")
    print()
    
    # Create embedding layer
    embedding_layer = TokenAndPositionEmbedding(
        max_length=max_length,
        vocabulary_size=vocab_size,
        embedding_dim=embedding_dim,
        positional_embedding_type="sinusoidal"
    )
    
    # Create dummy input: batch of 2 sequences, each with 4 tokens
    # Token IDs: 1, 5, 10, 15 (just random valid IDs)
    input_tokens = jnp.array([
        [1, 5, 10, 15, 0, 0, 0, 0],  # First sentence (padded to 8)
        [3, 8, 12, 18, 2, 0, 0, 0],  # Second sentence (padded to 8)
    ])
    
    print("Input tensor (token IDs):")
    trace_tensor("input_tokens", input_tokens, indent=1)
    
    # Forward pass
    embedded = embedding_layer(input_tokens)
    
    print("After TokenAndPositionEmbedding:")
    trace_tensor("embedded_output", embedded, indent=1)
    
    print("What just happened?")
    print("  1. Each token ID was looked up in an embedding matrix")
    print(f"     → Matrix shape: [{vocab_size} × {embedding_dim}]")
    print("  2. Sinusoidal position encoding was added (line 144-156 in layers.py)")
    print("  3. Result: context-aware starting representations")
    print()


def demo_attention_layer() -> None:
    """Demonstrates MultiHeadSelfAttention with shape tracing."""
    print("=" * 70)
    print("👁️ STEP 2: MULTI-HEAD SELF-ATTENTION")
    print("=" * 70)
    print()
    
    embedding_dim = 16
    num_heads = 4  # 4 attention heads
    
    print("Configuration:")
    print(f"  • Embedding dimension: {embedding_dim}")
    print(f"  • Number of heads: {num_heads}")
    print(f"  • Dimension per head: {embedding_dim // num_heads} (16 ÷ 4 = 4)")
    print()
    
    # Create attention layer
    attn_layer = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout_rate=0.0
    )
    
    # Dummy input: batch=1, seq_len=4, embed_dim=16
    # Pretend this is our embedded sentence
    dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, 4, embedding_dim))
    
    print("Input to attention layer:")
    trace_tensor("attention_input", dummy_input, indent=1)
    
    # Forward pass
    attn_output = attn_layer(dummy_input)
    
    print("After MultiHeadSelfAttention:")
    trace_tensor("attention_output", attn_output, indent=1)
    print()
    
    print("What just happened? (see layers.py:340-346)")
    print("  1. Input was projected into Query, Key, Value matrices")
    print(f"     • Q shape: [batch=1, seq=4, heads={num_heads}, head_dim=4]")
    print(f"     • K shape: [batch=1, seq=4, heads={num_heads}, head_dim=4]")
    print(f"     • V shape: [batch=1, seq=4, heads={num_heads}, head_dim=4]")
    print("  2. Attention scores computed: softmax(Q × K^T / sqrt(dim))")
    print("  3. Scores applied to V: weighted sum (our visualization from before!)")
    print("  4. Residual connection added: output + input")
    print("  5. Layer normalization applied")
    print("  6. Causal mask ensured we only look at previous tokens")
    print()


def demo_feedforward_layer() -> None:
    """Demonstrates FeedForwardNetwork with shape tracing."""
    print("=" * 70)
    print("🔄 STEP 3: FEED-FORWARD NETWORK")
    print("=" * 70)
    print()
    
    embedding_dim = 16
    mlp_dim = 64  # Expanded dimension (typically 4x embedding_dim)
    
    print("Configuration:")
    print(f"  • Input/Output dimension: {embedding_dim}")
    print(f"  • Hidden (MLP) dimension: {mlp_dim} (4× expansion)")
    print()
    
    # Create FFN layer
    ffn_layer = FeedForwardNetwork(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        dropout_rate=0.0,
        activation="relu"
    )
    
    # Same dummy input as before
    dummy_input = jax.random.normal(jax.random.PRNGKey(1), (1, 4, embedding_dim))
    
    print("Input to FFN:")
    trace_tensor("ffn_input", dummy_input, indent=1)
    
    # Forward pass
    ffn_output = ffn_layer(dummy_input)
    
    print("After FeedForwardNetwork:")
    trace_tensor("ffn_output", ffn_output, indent=1)
    print()
    
    print("What just happened? (see layers.py:279-293)")
    print(f"  1. Dense layer: {embedding_dim} → {mlp_dim} (expansion)")
    print("  2. ReLU activation: set negative values to 0")
    print(f"  3. Dense layer: {mlp_dim} → {embedding_dim} (projection back)")
    print("  4. Dropout (disabled for demo)")
    print("  5. Residual connection: output + input")
    print("  6. Layer normalization")
    print()
    print("Key insight: FFN processes each position independently,")
    print("while Attention mixes information ACROSS positions.")
    print("Together they create the magic!")
    print()


def demo_full_transformer_block() -> None:
    """Demonstrates complete TransformerBlock."""
    print("=" * 70)
    print("🏗️ STEP 4: COMPLETE TRANSFORMER BLOCK")
    print("=" * 70)
    print()
    
    embedding_dim = 16
    num_heads = 4
    mlp_dim = 64
    
    print("Configuration:")
    print(f"  • Embedding dim: {embedding_dim}")
    print(f"  • Attention heads: {num_heads}")
    print(f"  • MLP dim: {mlp_dim}")
    print()
    
    # Create full transformer block
    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout_rate=0.0,
        activation_function="relu"
    )
    
    # Input: embedded sentence
    dummy_input = jax.random.normal(jax.random.PRNGKey(2), (1, 4, embedding_dim))
    
    print("Input to TransformerBlock:")
    trace_tensor("block_input", dummy_input, indent=1)
    
    print("Block structure (layers.py:211-231):")
    print("  ┌─────────────────────────────────────┐")
    print("  │  Input: [1, 4, 16]                  │")
    print("  ↓                                     │")
    print("  ┌──────────────────┐                  │")
    print("  │ MultiHeadAttention│ ←──── Mix info   │")
    print("  │ + Residual        │                  │")
    print("  │ + LayerNorm       │                  │")
    print("  └──────────────────┘                  │")
    print("  ↓                                     │")
    print("  ┌──────────────────┐                  │")
    print("  │ FeedForward       │ ←──── Process    │")
    print("  │ + Residual        │                  │")
    print("  │ + LayerNorm       │                  │")
    print("  └──────────────────┘                  │")
    print("  ↓                                     │")
    print("  Output: [1, 4, 16] ←──────────────────┘")
    print()
    
    # Forward pass
    block_output = block(dummy_input)
    
    print("After TransformerBlock:")
    trace_tensor("block_output", block_output, indent=1)
    print()


def demo_full_model() -> None:
    """Demonstrates the complete model creation using Keras functional API."""
    print("=" * 70)
    print("🚀 STEP 5: COMPLETE MODEL (Building with Your Layers)")
    print("=" * 70)
    print()
    
    # Tiny model configuration
    vocab_size = 50
    max_length = 16
    embedding_dim = 32
    mlp_dim = 128
    num_heads = 4
    num_blocks = 2
    
    print("Configuration:")
    print(f"  • Vocabulary: {vocab_size} tokens")
    print(f"  • Max sequence: {max_length} tokens")
    print(f"  • Embedding: {embedding_dim} dimensions")
    print(f"  • MLP hidden: {mlp_dim} dimensions")
    print(f"  • Attention heads: {num_heads}")
    print(f"  • Transformer blocks: {num_blocks} (stacked)")
    print()
    
    # Build model using Keras functional API with your layers
    inputs = layers.Input(shape=(max_length,), dtype="int32")
    
    # Your embedding layer
    x = TokenAndPositionEmbedding(
        max_length=max_length,
        vocabulary_size=vocab_size,
        embedding_dim=embedding_dim
    )(inputs)
    
    print("Model architecture:")
    print(f"  Input:          {inputs.shape}")
    print(f"  ↓")
    print(f"  Embedding:      (batch, {max_length}, {embedding_dim})")
    
    # Stack your transformer blocks
    for i in range(num_blocks):
        x = TransformerBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=0.0
        )(x)
        print(f"  ↓")
        print(f"  TransformerBlock {i+1}: (batch, {max_length}, {embedding_dim})")
    
    # Output layer
    outputs = layers.Dense(vocab_size)(x)
    print(f"  ↓")
    print(f"  Dense:          (batch, {max_length}, {vocab_size}) ← logits!")
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print()
    print("Model created successfully! ✅")
    print()
    
    # Show parameter count
    model.summary()
    print()
    
    # Test forward pass
    dummy_input = jnp.ones((2, max_length), dtype=jnp.int32)
    print("Test input shape:", dummy_input.shape)
    
    output = model(dummy_input, training=False)
    print("Output shape:", output.shape)
    print("Output represents: logits for next token prediction")
    print()


def main() -> None:
    """Main execution for Week 1 model tracing."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "WEEK 1: MODEL TRACE" + " " * 27 + "║")
    print("║" + " " * 15 + "See Real Tensors Flow Through Your Code" + " " * 14 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    print("This script traces tensor shapes through each layer of your")
    print("actual codebase. It's like adding print() statements everywhere!")
    print()
    
    # Run all demonstrations
    demo_embeddings()
    demo_attention_layer()
    demo_feedforward_layer()
    demo_full_transformer_block()
    demo_full_model()
    
    print("=" * 70)
    print("🎯 KEY TAKEAWAYS FROM WEEK 1")
    print("=" * 70)
    print()
    print("1. TENSOR SHAPES tell the story:")
    print("   Input:  [batch, seq_len]          ← token IDs")
    print("   Embed:  [batch, seq_len, dim]    ← vectors")
    print("   Attn:   [batch, seq_len, dim]    ← mixed vectors")
    print("   FFN:    [batch, seq_len, dim]    ← processed vectors")
    print("   Output: [batch, seq_len, vocab]  ← predictions")
    print()
    print("2. RESIDUAL CONNECTIONS (skip connections):")
    print("   → Layer output = Layer(input) + input")
    print("   → Why? Helps gradients flow, prevents vanishing gradients")
    print("   → See layers.py:291 and 345")
    print()
    print("3. LAYER NORMALIZATION:")
    print("   → Normalizes across the embedding dimension")
    print("   → Why? Stabilizes training, prevents exploding values")
    print()
    print("4. YOUR CODE IS FUNCTIONAL! 🎉")
    print("   You have a working transformer implementation.")
    print("   The journey now is understanding every line.")
    print()
    print("=" * 70)
    print()
    print("Next: Study the visualization (week1_attention_visualization.py)")
    print("      then come back and trace through this with debugger!")
    print()


if __name__ == "__main__":
    main()
