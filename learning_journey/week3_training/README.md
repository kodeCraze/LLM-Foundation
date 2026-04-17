# Week 3: Training Loop & The Forward Pass ✅

**Focus:** Understanding how LLMs learn through gradient descent

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand the 4-step training loop (Forward → Loss → Backward → Update)
- ✅ See gradient descent in action on a toy dataset
- ✅ Understand cross-entropy loss for language modeling
- ✅ Build intuition for why training works
- ✅ Connect to your `training/` directory

---

## Week 3 Scripts

### 1. `week3_training_loop.py`
**Purpose:** Train a tiny transformer on synthetic data

**What you'll learn:**
- The complete training loop
- How loss decreases over time
- Why overfitting small data is essential
- How to debug training issues

**Run it:**
```bash
cd learning_journey/week3_training
python week3_training_loop.py
```

**Output:**
- Console: Step-by-step training process
- `training_progress.png`: Loss and accuracy curves
- Final predictions vs targets

---

## Key Concepts

### The 4-Step Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. FORWARD PASS                                            │
│     Input: [1, 2, 3, 4, 1, 2, 3, 4]                        │
│       ↓                                                     │
│     Model: Embedding → Transformer → Dense                 │
│       ↓                                                     │
│     Output: logits [batch, seq_len, vocab_size]             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  2. LOSS CALCULATION                                        │
│     Target: [2, 3, 4, 1, 2, 3, 4, 1]                       │
│     Prediction: softmax(logits) → probabilities            │
│     Loss: -log(prob of correct token)                     │
│     → "How surprised is the model by the correct answer?"   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  3. BACKWARD PASS (BACKPROPAGATION)                         │
│     Question: Which weights caused the error?               │
│     Answer: Calculate gradients ∂Loss/∂Weight             │
│     → Chain rule applied through all layers               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  4. WEIGHT UPDATE (GRADIENT DESCENT)                        │
│     weight_new = weight_old - lr × gradient                 │
│     → Move in direction that reduces loss                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔁 REPEAT FOR THOUSANDS OF STEPS                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Entropy Loss Explained

```
CrossEntropy = -log(probability_assigned_to_correct_token)

Examples:
  Model predicts correct token with 100% confidence:
    → loss = -log(1.0) = 0 ✅
    
  Model predicts correct token with 50% confidence:
    → loss = -log(0.5) = 0.69 ⚠️
    
  Model predicts correct token with 1% confidence:
    → loss = -log(0.01) = 4.6 ❌
```

**Key insight:** Loss heavily penalizes confident wrong predictions!

### Gradient Descent Intuition

Imagine you're on a mountain (loss landscape) in fog:
- **You want to:** Get to the lowest valley (minimum loss)
- **You can see:** Only your immediate surroundings (gradient)
- **Gradient tells you:** Which direction is downhill
- **Learning rate:** Size of your step
  - Too small: Takes forever
  - Too large: Might overshoot and fall
  - Just right: Steady descent

### Why Overfit Small Data First?

**The Principle:** If your model can't memorize 5 examples, it won't learn millions.

**Benefits:**
1. **Fast feedback** - Train in seconds, not hours
2. **Debug issues** - Verify model architecture is correct
3. **Understand capacity** - See if model has enough parameters
4. **Build intuition** - Watch loss actually decrease

**This week's demo:**
- 4 training sequences
- 2 transformer blocks
- 100 epochs
- Should achieve 100% accuracy!

---

## Week 3 Exercises

### Exercise 1: Modify the Toy Dataset
```python
# Create your own patterns
sequences = [
    [1, 1, 2, 2, 1, 1, 2, 2],  # Your pattern
    [1, 1, 2, 2, 1, 1, 2, 2],  # Target (shifted)
]

# Questions:
# 1. Can the model learn it?
# 2. How many epochs needed?
# 3. What if you make it more complex?
```

### Exercise 2: Vary Model Size
```python
# Try different configurations
configs = [
    {'embedding_dim': 16, 'num_blocks': 1},   # Tiny
    {'embedding_dim': 32, 'num_blocks': 2},   # Small
    {'embedding_dim': 64, 'num_blocks': 4},   # Medium
]

# Questions:
# 1. Which learns fastest?
# 2. Which overfits most?
# 3. What's the tradeoff?
```

### Exercise 3: Explore Your Training Directory
```bash
# List training files
ls training/

# Read the loss implementation
cat training/losses.py

# Read the callbacks
cat training/callbacks.py

# Read the model builder
cat training/model.py
```

**Questions:**
1. How does CustomMaskPadLoss handle padding?
2. What callbacks does your codebase have?
3. How does create_model() differ from our demo?

### Exercise 4: Implement a Simple Callback
```python
class LossPrinter(keras.callbacks.Callback):
    """Print loss every N batches."""
    
    def __init__(self, print_every=10):
        self.print_every = print_every
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.print_every == 0:
            print(f"Batch {batch}: loss={logs['loss']:.4f}")

# Use it:
demo.model.fit(X, y, epochs=100, callbacks=[LossPrinter()])
```

---

## Connection to Your Codebase

Your `training/` directory:

```
training/
├── __init__.py          # Exports: create_model, CustomMaskPadLoss, etc.
├── callbacks.py         # CustomAccuracyPrinter, TextGenerator
├── losses.py            # CustomMaskPadLoss (handles padding)
└── model.py             # create_model() function
```

**Compare with our demo:**
- Our demo: Minimal implementation for understanding
- Your code: Production-ready with more features

**Key differences:**
1. Your `create_model()` supports more optimizers
2. Your `CustomMaskPadLoss` properly masks padding
3. Your callbacks generate text during training
4. Your code has better error handling

---

## Week 3 Completion Checklist

- [ ] Ran `week3_training_loop.py` and watched loss decrease
- [ ] Saw training curves in `training_progress.png`
- [ ] Model achieved >95% accuracy on toy dataset
- [ ] Can explain the 4-step training loop
- [ ] Understand why cross-entropy loss makes sense
- [ ] Explored your `training/` directory
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Training Loop
1. **Forward pass:** Input → Model → Predictions
2. **Loss calculation:** How wrong are we?
3. **Backward pass:** Calculate gradients via backprop
4. **Update:** Adjust weights to reduce loss

### Loss Functions
1. **Cross-entropy:** -log(prob of correct answer)
2. **Perplexity:** exp(cross_entropy) - "effective vocab size"
3. **Masked loss:** Ignore padding tokens

### Optimization
1. **Gradient descent:** Walk downhill on loss landscape
2. **Learning rate:** Step size (crucial hyperparameter)
3. **Overfitting small data:** Essential debugging tool

---

## Bridge to Week 4

Next week: **Optimization Deep Dive**
- Adam optimizer (not just SGD)
- Learning rate schedules
- Gradient clipping
- Weight decay

---

## Resources

### Reading
- "Gradient Descent Explained" (any good ML tutorial)
- "Understanding Binary Cross-Entropy"
- Your `training/losses.py` implementation

### Interactive
- [Loss Landscape Visualization](https://losslandscape.com) (if available)
- Play with learning rates in our demo

### Code Study
- `training/model.py` - How your codebase builds models
- `training/losses.py` - Loss computation details
- `training/callbacks.py` - Training monitoring

---

## Week 3 Status

**Your progress:**
- ✅ Training loop understanding
- ✅ Cross-entropy loss explained
- ✅ Gradient descent intuition
- ✅ Practical overfitting demonstration
- ✅ Connection to your codebase

**Ready for Week 4?**

If yes → See `../week4_optimization/`

If no → Modify the demo, break it, fix it, learn!

---

*Remember: Training is just iterative loss minimization. Simple idea, powerful results!*
