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

"""Week 10: Post-Training - RLHF and DPO.

This script demonstrates:
1. Supervised Fine-Tuning (SFT)
2. RLHF (Reinforcement Learning from Human Feedback)
3. PPO (Proximal Policy Optimization)
4. DPO (Direct Preference Optimization) - The modern alternative

How to align LLMs with human preferences.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys

sys.path.insert(0, 'd:\\Deepmind_Reserch\\ai_foundations')


class PostTrainingPipeline:
    """Demonstrates the post-training pipeline."""
    
    def explain_sft(self) -> None:
        """Explain Supervised Fine-Tuning."""
        print("=" * 70)
        print("🎓 STEP 1: SUPERVISED FINE-TUNING (SFT)")
        print("=" * 70)
        print()
        
        print("THE GOAL:")
        print("  Teach the pretrained model to follow instructions")
        print("  and engage in helpful conversations")
        print()
        
        print("THE PROCESS:")
        print("  1. Start with pretrained model (already knows language)")
        print("  2. Create instruction-following dataset")
        print("  3. Train model to generate appropriate responses")
        print()
        
        print("SFT DATASET EXAMPLE:")
        print()
        print("  Instruction: 'Explain quantum computing to a 10-year-old'")
        print("  Response: 'Quantum computing is like having a magical coin that...'")
        print()
        print("  Instruction: 'Write a Python function to sort a list'")
        print("  Response: 'Here's a simple Python function using built-in sort...'")
        print()
        print("  Instruction: 'Is it safe to drink expired milk?'")
        print("  Response: 'No, drinking expired milk can make you sick...'")
        print()
        
        print("TRAINING:")
        print("  Same as pretraining, but:")
        print("    • Smaller learning rate (1e-5 to 1e-6)")
        print("    • Shorter training (1-10 epochs)")
        print("    • Higher quality, curated data")
        print("    • Usually freeze most layers, tune only top layers")
        print()
        
        print("RESULT:")
        print("  Model can follow instructions but may still be:")
        print("    • Toxic or harmful")
        print("    • Unhelpful or verbose")
        print("    • Uncertain about what humans prefer")
        print()
        print("  → Need RLHF to align with human preferences")
        print()
    
    def explain_rlhf(self) -> None:
        """Explain RLHF conceptually."""
        print("=" * 70)
        print("🎯 STEP 2: RLHF (REINFORCEMENT LEARNING FROM HUMAN FEEDBACK)")
        print("=" * 70)
        print()
        
        print("THE PROBLEM:")
        print("  SFT model can generate responses, but we want it to:")
        print("    • Be helpful, harmless, and honest")
        print("    • Match human preferences")
        print("    • Not be toxic or biased")
        print()
        
        print("THE SOLUTION - THREE STEPS:")
        print()
        
        print("  STEP 2A: Train Reward Model")
        print("  ┌──────────────────────────────────────────────────────┐")
        print("  │  1. Humans compare model outputs                     │")
        print("  │     Response A vs Response B                         │")
        print("  │                                                      │")
        print("  │  2. Train Reward Model to predict human preference   │")
        print("  │     Input: (prompt, response)                        │")
        print("  │     Output: score (how good is this response?)     │")
        print("  │                                                      │")
        print("  │  3. Reward Model learns human values                 │")
        print("  └──────────────────────────────────────────────────────┘")
        print()
        
        print("  STEP 2B: RL Training with PPO")
        print("  ┌──────────────────────────────────────────────────────┐")
        print("  │  Loop:                                               │")
        print("  │    1. Language model generates response              │")
        print("  │    2. Reward Model scores the response               │")
        print("  │    3. PPO updates LM to maximize reward              │")
        print("  │    4. KL penalty prevents model from drifting        │")
        print("  │                                                      │")
        print("  │  Result: LM learns to generate high-reward responses │")
        print("  └──────────────────────────────────────────────────────┘")
        print()
        
        print("VISUALIZATION:")
        print()
        print("  Pretrained → SFT → RLHF → Aligned Model")
        print("     │          │       │")
        print("     │          │       └─► Human preferences")
        print("     │          │           (reward model)")
        print("     │          └─► Instruction following")
        print("     │              (supervised data)")
        print("     └─► Language understanding")
        print("         (pretraining corpus)")
        print()
    
    def explain_ppo(self) -> None:
        """Explain PPO algorithm conceptually."""
        print("=" * 70)
        print("⚙️  PPO (PROXIMAL POLICY OPTIMIZATION)")
        print("=" * 70)
        print()
        
        print("THE CHALLENGE:")
        print("  Training LLMs with RL is unstable!")
        print("  Small policy changes → big output changes")
        print("  Can collapse to gibberish quickly")
        print()
        
        print("PPO SOLUTION:")
        print("  1. Don't update too much at once")
        print("  2. Clip the policy update to prevent large changes")
        print("  3. Add KL penalty to stay close to reference (SFT) model")
        print()
        
        print("PPO OBJECTIVE:")
        print()
        print("  L = E[ min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A) ] - β × KL")
        print()
        print("  Where:")
        print("    r(θ) = π_θ(a|s) / π_old(a|s)  [probability ratio]")
        print("    A = advantage (how much better than average)")
        print("    ε = clip parameter (usually 0.2)")
        print("    β = KL penalty coefficient")
        print()
        
        print("INTUITION:")
        print("  • If new policy is better (A > 0): allow update")
        print("  • If new policy is worse (A < 0): discourage update")
        print("  • But CLIP: don't change probability by more than 20%")
        print("  • And KL penalty: don't drift too far from SFT model")
        print()
        
        print("VISUALIZATION:")
        print()
        print("  Reward")
        print("    │")
        print("  10├─────  🏆 High Reward (good response)")
        print("    │     ╱")
        print("   5├────╱── 📝 SFT Baseline")
        print("    │   ╱")
        print("   0├───╱──── 💩 Low Reward (bad response)")
        print("    │  ╱")
        print("  -5├──╱")
        print("    │")
        print("    └───────────────────────")
        print("       RL training →")
        print()
        print("  PPO pushes model toward high-reward region")
        print("  But gently, with constraints")
        print()
    
    def explain_dpo(self) -> None:
        """Explain DPO (Direct Preference Optimization)."""
        print("=" * 70)
        print("🚀 DPO (DIRECT PREFERENCE OPTIMIZATION)")
        print("=" * 70)
        print()
        
        print("THE PROBLEM WITH RLHF+PPO:")
        print("  ❌ Complex (need reward model + PPO + reference model)")
        print("  ❌ Unstable (hyperparameter sensitive)")
        print("  ❌ Memory hungry (4 models in GPU)")
        print("  ❌ Slow (many forward/backward passes)")
        print()
        
        print("DPO SOLUTION:")
        print("  Skip the reward model and PPO entirely!")
        print("  Train directly on human preferences")
        print()
        
        print("HOW IT WORKS:")
        print()
        print("  Given: (prompt, winning_response, losing_response)")
        print()
        print("  DPO Loss:")
        print("    L = -log σ(β × log[π_win/π_ref] - β × log[π_lose/π_ref])")
        print()
        print("  Simplified:")
        print("    • Increase probability of winning response")
        print("    • Decrease probability of losing response")
        print("    • β controls how far from reference (SFT) model")
        print()
        
        print("ADVANTAGES:")
        print("  ✅ Simple (just one model to train)")
        print("  ✅ Stable (no RL instability)")
        print("  ✅ Fast (2× faster than PPO)")
        print("  ✅ Memory efficient (no reward model)")
        print("  ✅ Often better results!")
        print()
        
        print("COMPARISON:")
        print()
        print("  RLHF+PPO:")
        print("    Models: Policy + Reference + Reward + Critic (4!)")
        print("    Steps: Generate → Score → PPO update → Repeat")
        print("    Memory: 4× model size")
        print()
        print("  DPO:")
        print("    Models: Policy + Reference (2)")
        print("    Steps: Forward pass on (win, lose) → Update")
        print("    Memory: 2× model size")
        print()
        
        print("ADOPTION:")
        print("  • Zephyr (Hugging Face): DPO")
        print("  • Tülu (AllenAI): DPO")
        print("  • Many 2024 models: DPO over RLHF+PPO")
        print("  • Llama 2: Actually used RLHF (not DPO)")
        print()
    
    def compare_methods(self) -> None:
        """Compare RLHF vs DPO."""
        print("=" * 70)
        print("⚖️  RLHF+PPO vs DPO COMPARISON")
        print("=" * 70)
        print()
        
        print("┌──────────────────┬─────────────────┬─────────────────┐")
        print("│ Aspect           │ RLHF + PPO      │ DPO             │")
        print("├──────────────────┼─────────────────┼─────────────────┤")
        print("│ Complexity       │ High            │ Low             │")
        print("│ Number of models │ 4               │ 2               │")
        print("│ Stability        │ Can be unstable │ Very stable     │")
        print("│ Training speed   │ Slow            │ Fast (2×)       │")
        print("│ Memory usage     │ High (4×)       │ Lower (2×)      │")
        print("│ Reward model     │ Required        │ Not needed      │")
        print("│ RL algorithm     │ PPO             │ None (direct)   │")
        print("│ Performance      │ Good            │ Often better    │")
        print("│ Adoption (2024)  │ Decreasing      │ Increasing      │")
        print("└──────────────────┴─────────────────┴─────────────────┘")
        print()
        
        print("RECOMMENDATION:")
        print("  • For research/production: DPO (simpler, better)")
        print("  • For learning: Understand both (RLHF is foundational)")
        print("  • Some teams still use RLHF (more control over reward)")
        print()
    
    def visualize_training_process(self) -> None:
        """Visualize the training process."""
        print("=" * 70)
        print("📈 TRAINING PROCESS VISUALIZATION")
        print("=" * 70)
        print()
        
        # Simulate training curves
        np.random.seed(42)
        steps = np.arange(1000)
        
        # Pretraining (very long, steady decrease)
        pretrain_loss = 4.0 * np.exp(-steps / 500) + 2.0
        pretrain_loss += np.random.randn(1000) * 0.05
        
        # SFT (quick adaptation)
        sft_loss = 2.5 * np.exp(-steps / 200) + 1.5
        sft_loss += np.random.randn(1000) * 0.03
        
        # RLHF/DPO (optimization toward preferences)
        rlhf_reward = 5.0 * (1 - np.exp(-steps / 300)) + 2.0
        rlhf_reward += np.random.randn(1000) * 0.1
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Pretraining
        ax1 = axes[0]
        ax1.plot(steps, pretrain_loss, 'b-', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Training Steps (Millions)')
        ax1.set_ylabel('Loss')
        ax1.set_title('Phase 1: Pretraining\n(Learn language)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(1.5, 5)
        
        # SFT
        ax2 = axes[1]
        ax2.plot(steps, sft_loss, 'g-', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Training Steps (Thousands)')
        ax2.set_ylabel('Loss')
        ax2.set_title('Phase 2: SFT\n(Instruction following)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(1.3, 2.8)
        
        # RLHF/DPO
        ax3 = axes[2]
        ax3.plot(steps, rlhf_reward, 'r-', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Training Steps (Thousands)')
        ax3.set_ylabel('Reward')
        ax3.set_title('Phase 3: RLHF/DPO\n(Human preference)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(2, 8)
        
        plt.tight_layout()
        plt.savefig('post_training_phases.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to 'post_training_phases.png'")
        print()
        
        print("OBSERVATIONS:")
        print("  1. Pretraining: Long, steady, high initial loss")
        print("  2. SFT: Quick adaptation to instruction format")
        print("  3. RLHF/DPO: Reward increases as model aligns")
        print()


class PreferenceDatasetExamples:
    """Show examples of preference data."""
    
    def show_examples(self) -> None:
        """Show preference dataset examples."""
        print("=" * 70)
        print("📝 PREFERENCE DATA EXAMPLES")
        print("=" * 70)
        print()
        
        examples = [
            {
                'prompt': 'How can I hack into someone\'s email?',
                'win': 'I can\'t help with that. Hacking into someone\'s email '
                       'is illegal and unethical. If you\'re concerned about '
                       'your own account security, I can help with that.',
                'lose': 'Here are some common methods hackers use to breach '
                        'email accounts: 1) Phishing emails... [harmful detailed guide]',
                'reason': 'Safety - refuse harmful requests',
            },
            {
                'prompt': 'Explain quantum physics',
                'win': 'Quantum physics is the study of matter and energy at '
                       'the smallest scales. At this level, particles behave '
                       'like waves and can exist in multiple states simultaneously. '
                       'Key concepts include superposition, entanglement, and '
                       'the uncertainty principle.',
                'lose': 'Quantum physics is complicated. It involves particles '
                        'and stuff. You should read a textbook if you want to '
                        'learn about it.',
                'reason': 'Helpfulness - comprehensive vs dismissive',
            },
            {
                'prompt': 'Write a poem about the ocean',
                'win': 'Blue depths calling, waves embrace the shore,\n'
                       'Salt and spray, ancient rhythms roar.\n'
                       'Moon pulls tides in endless dance,\n'
                       'Mysteries deep in liquid trance.',
                'lose': 'The ocean is big and blue. Fish live there. '
                        'Waves crash. The end.',
                'reason': 'Quality - creative vs generic',
            },
            {
                'prompt': 'What\'s 2+2?',
                'win': '2+2 equals 4.',
                'lose': '2+2 equals 4. Let me explain addition in detail. '
                        'Addition is a mathematical operation that represents '
                        'the total amount of objects combined in a collection. '
                        'When we add 2 and 2, we take two items and combine them '
                        'with two more items, resulting in four items total... '
                        '[verbose unnecessary explanation]',
                'reason': 'Conciseness - direct vs verbose',
            },
        ]
        
        for i, ex in enumerate(examples, 1):
            print(f"EXAMPLE {i}: {ex['reason']}")
            print(f"Prompt: {ex['prompt']}")
            print()
            print("✅ Winning response (chosen):")
            print(f"  {ex['win'][:100]}...")
            print()
            print("❌ Losing response (rejected):")
            print(f"  {ex['lose'][:100]}...")
            print()
            print("-" * 50)
            print()


def main():
    """Main execution for Week 10."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "WEEK 10: POST-TRAINING" + " " * 26 + "║")
    print("║" + " " * 16 + "RLHF, DPO & Model Alignment" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Post-training pipeline
    pipeline = PostTrainingPipeline()
    pipeline.explain_sft()
    pipeline.explain_rlhf()
    pipeline.explain_ppo()
    pipeline.explain_dpo()
    pipeline.compare_methods()
    pipeline.visualize_training_process()
    
    # Preference examples
    prefs = PreferenceDatasetExamples()
    prefs.show_examples()
    
    print("=" * 70)
    print("🎯 WEEK 10 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. THREE-STEP PIPELINE:")
    print("   • Pretrain: Learn language from corpus")
    print("   • SFT: Learn to follow instructions")
    print("   • RLHF/DPO: Align with human preferences")
    print()
    print("2. RLHF COMPONENTS:")
    print("   • Reward Model: Predicts human preferences")
    print("   • PPO: RL algorithm to optimize policy")
    print("   • KL Penalty: Prevents drift from SFT")
    print()
    print("3. DPO (MODERN ALTERNATIVE):")
    print("   • Directly trains on preferences")
    print("   • No reward model needed")
    print("   • Simpler, faster, often better")
    print("   • 2024 recommendation: Use DPO")
    print()
    print("4. HUMAN FEEDBACK:")
    print("   • Comparisons: A vs B (which is better?)")
    print("   • Criteria: Helpful, harmless, honest")
    print("   • Quality > Quantity (careful labeling)")
    print()
    print("5. SAFETY:")
    print("   • Models can refuse harmful requests")
    print("   • Can admit uncertainty")
    print("   • Less toxic, more helpful")
    print()
    print("=" * 70)
    print()
    print("Next: Week 11 - Inference Optimization")
    print("      (KV-cache, quantization, speculative decoding)")
    print()


if __name__ == "__main__":
    main()
