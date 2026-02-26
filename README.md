# Tiny Reasoner

A small reasoning model project combining ideas from [TinyLlama](https://github.com/jzhang38/TinyLlama) and [RWKV-LM](https://github.com/BlinkDL/RWKV-LM).

## Features

- **Two Architectures**: Transformer (TinyLlama-style) and RWKV-7 (RNN-based)
- **Efficient Training**: LoRA, 8-bit optimizers, gradient checkpointing
- **Reasoning Focus**: Chain-of-thought data preparation and evaluation
- **Comprehensive Benchmarking**: GSM8K, perplexity, and custom benchmarks

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test Installation

```bash
python test_models.py
```

### 2. Prepare Data

```bash
python train.py --prepare-data
```

### 3. Train a Model

```bash
# Small transformer (recommended for 8GB GPU)
python train.py --config micro --batch-size 2 --gradient-accumulation 8 --max-steps 5000

# With LoRA (more efficient)
python train.py --config tiny --use-lora --lora-r 8 --batch-size 4 --max-steps 10000

# RWKV-7 model
python train.py --config tiny-rwkv --batch-size 4 --max-steps 10000
```

### 4. Run Inference

```bash
# Chat mode
python inference.py --checkpoint checkpoints/checkpoint-1000/model.pt --config tiny --mode chat

# Reasoning mode
python inference.py --checkpoint checkpoints/checkpoint-1000/model.pt --config tiny --mode reasoning

# Single prompt
python inference.py --checkpoint checkpoints/checkpoint-1000/model.pt --config tiny --mode generate --prompt "What is 2+2?"
```

### 5. Evaluate

```bash
python train.py --config tiny --eval-only --checkpoint checkpoints/checkpoint-1000/model.pt
```

## Model Configurations

| Config | Architecture | Layers | Embed Dim | Params | VRAM (Training) |
|--------|-------------|--------|-----------|--------|-----------------|
| micro | Transformer | 6 | 384 | ~25M | ~2GB |
| tiny | Transformer | 8 | 512 | ~50M | ~4GB |
| small | Transformer | 12 | 768 | ~125M | ~8GB |
| tiny-rwkv | RWKV-7 | 12 | 512 | ~50M | ~3GB |
| small-rwkv | RWKV-7 | 18 | 768 | ~125M | ~6GB |

## Architecture Comparison

### Transformer (TinyLlama-style)
- Grouped Query Attention (GQA)
- Rotary Position Embeddings
- SwiGLU FFN
- RMSNorm

### RWKV-7
- Linear time, constant space
- No KV-cache needed
- Efficient inference
- Better for limited VRAM

## Training Tips for 8GB GPU

1. **Use smaller batch sizes with gradient accumulation**:
   ```bash
   --batch-size 2 --gradient-accumulation 8
   ```

2. **Enable 8-bit optimizer**:
   ```bash
   --use-8bit
   ```

3. **Use LoRA for efficient fine-tuning**:
   ```bash
   --use-lora --lora-r 8
   ```

4. **Start with smaller model**:
   ```bash
   --config micro
   ```

## Benchmarking

Run the evaluation suite:

```python
from benchmarks import BenchmarkSuite, GSM8KBenchmark, PerplexityBenchmark
from models import create_model
from transformers import AutoTokenizer

model = create_model("tiny")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

suite = BenchmarkSuite()
suite.add_benchmark(GSM8KBenchmark())
suite.add_benchmark(PerplexityBenchmark())

results = suite.run_all(model, tokenizer)
print(suite.generate_report(results))
```

## Project Structure

```
tinyllama/
├── config.py           # Model configurations
├── models/
│   ├── transformer.py  # Transformer implementation
│   └── rwkv7.py        # RWKV-7 implementation
├── training/
│   └── trainer.py      # Training loop with LoRA support
├── data/
│   └── prepare_data.py # Data preparation utilities
├── benchmarks/
│   └── eval.py         # Evaluation suite
├── train.py            # Training script
├── inference.py        # Inference script
└── test_models.py      # Model tests
```

## Next Steps

1. **Scale up**: Once you get another GPU, try `--config small` for better results
2. **Fine-tune**: Use LoRA adapters for specific reasoning tasks
3. **Curriculum**: Train on easy examples first, then harder ones
4. **Distill**: Use larger models to generate training data

## References

- [TinyLlama](https://github.com/jzhang38/TinyLlama) - 1.1B Llama model training
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) - RWKV architecture
- [GSM8K](https://github.com/openai/grade-school-math) - Math reasoning dataset
