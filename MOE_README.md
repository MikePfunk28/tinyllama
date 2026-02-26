# MoE Project Summary

## Architecture

**Base Model:** Gemma-3-270M (436M params)
**Approach:** Shared base model with domain-specific LoRA adapters
**Routing:** Keyword-based expert selection

## Trained LoRA Adapters

| Domain | Val Loss | Training Samples | Status |
|--------|----------|------------------|--------|
| math | 1.23 | 500 | ✅ Trained |
| reasoning | 1.84 | 500 | ✅ Trained |
| code | 1.57 | 500 | ✅ Trained |
| conversation | 2.45 | 500 | ✅ Trained |
| knowledge | 1.74 | 500 | ✅ Trained |

## Files

```
M:/tinyllama/
├── moe_efficient.py          # Main MoE with shared base + LoRA
├── moe_continuous.py          # Continuous training loop
├── fast_train.py              # Scratch training (older approach)
├── hierarchical_moe.py        # Hierarchical MoE (older approach)
│
├── experts/
│   ├── math_lora_adapter.pt
│   ├── reasoning_lora_adapter.pt
│   ├── code_lora_adapter.pt
│   ├── conversation_lora_adapter.pt
│   └── knowledge_lora_adapter.pt
│
├── validation_results/        # Training histories
└── moe_training.log          # Training logs
```

## Usage

### Train LoRA Adapters
```bash
# Train single domain
python moe_efficient.py --train math --epochs 1

# Train all domains
python moe_efficient.py --train math reasoning code conversation knowledge --epochs 1

# Continuous training
python moe_continuous.py
```

### Test MoE
```bash
python moe_efficient.py --test
```

## Routing Logic

The MoE routes queries based on keywords:

- **math**: calculate, math, +, -, *, /, =
- **code**: code, python, function, program
- **knowledge**: explain, what is, why, how
- **conversation**: hello, hi, how are, help
- **reasoning**: default for complex problems

## Sample Outputs

```
Q: What is 15 + 27?
Routing to: math
A: 15 + 27 = 42

Q: Write a Python function to reverse a string.
Routing to: code
A: def reverse_string(s):
    return s[::-1]

Q: Solve: x + 5 = 12
Routing to: math
A: x = 12 - 5 = 7
```

## Next Steps

1. **Improve Routing:** Train a neural router instead of keyword matching
2. **More Training Data:** Increase from 500 to 2000+ samples per domain
3. **More Epochs:** Train for 3-5 epochs for better convergence
4. **Larger LoRA Rank:** Increase from 4 to 8 or 16
5. **Better Base Model:** Try Gemma-3-1B-IT for better quality
6. **Merge Adapters:** Combine LoRA adapters for inference

## Performance

- **Training Speed:** ~3.5 it/s on AMD RX6600
- **Inference Speed:** ~20 tokens/sec
- **Memory Usage:** ~2GB GPU (base model only)
- **LoRA Adapter Size:** ~2MB each

## External Resources

- Datasets: HuggingFace (DAPO-Math-17k, GSM8K, UltraFeedback, etc.)
- Base Model: google/gemma-3-270m
- Framework: PyTorch + DirectML
