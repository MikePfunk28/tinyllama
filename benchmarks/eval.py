import torch
import json
import re
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
from dataclasses import dataclass
import time


@dataclass
class BenchmarkResult:
    name: str
    accuracy: float
    total_samples: int
    correct: int
    avg_latency_ms: float
    details: Optional[Dict] = None


class BaseBenchmark:
    def __init__(self, name: str):
        self.name = name
        self.samples = []

    def load_data(self, path: Optional[str] = None):
        raise NotImplementedError

    def evaluate(self, model, device: str = "cuda") -> BenchmarkResult:
        raise NotImplementedError

    def format_prompt(self, sample: Dict) -> str:
        raise NotImplementedError

    def parse_response(self, response: str) -> str:
        raise NotImplementedError


class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, data_path: str = "./data/gsm8k_test.jsonl"):
        super().__init__("GSM8K")
        self.data_path = data_path

    def load_data(self, path: Optional[str] = None):
        path = path or self.data_path
        self.samples = []

        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(
                    {
                        "question": item["question"],
                        "answer": self._extract_answer(item["answer"]),
                    }
                )

    def _extract_answer(self, answer_text: str) -> str:
        match = re.search(r"####\s*(.+)$", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_text.strip()

    def format_prompt(self, sample: Dict) -> str:
        return f"Question: {sample['question']}\n\nLet's think step by step:\n"

    def parse_response(self, response: str) -> str:
        patterns = [
            r"(?:answer|result|solution)\s*(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)",
            r"####\s*([+-]?\d+(?:\.\d+)?)",
            r"\b([+-]?\d+(?:\.\d+)?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")
        return ""

    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        if not self.samples:
            self.load_data()

        samples = self.samples[:num_samples] if num_samples else self.samples
        correct = 0
        latencies = []

        model.eval()
        for sample in tqdm(samples, desc=f"Evaluating {self.name}"):
            prompt = self.format_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            start = time.time()
            with torch.no_grad():
                if hasattr(model, "generate"):
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 1.0,
                        top_k=40,
                    )
                else:
                    outputs = inputs["input_ids"]

            latency = (time.time() - start) * 1000
            latencies.append(latency)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted = self.parse_response(response)
            expected = sample["answer"]

            if self._compare_answers(predicted, expected):
                correct += 1

        accuracy = correct / len(samples) * 100
        avg_latency = sum(latencies) / len(latencies)

        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            total_samples=len(samples),
            correct=correct,
            avg_latency_ms=avg_latency,
        )

    def _compare_answers(self, predicted: str, expected: str) -> bool:
        try:
            p = float(predicted.strip().replace(",", ""))
            e = float(expected.strip().replace(",", ""))
            return abs(p - e) < 0.01
        except (ValueError, AttributeError):
            return predicted.strip().lower() == expected.strip().lower()


class PerplexityBenchmark(BaseBenchmark):
    def __init__(self, name: str = "Perplexity"):
        super().__init__(name)

    def load_data(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()

    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 2048,
        stride: int = 512,
    ) -> BenchmarkResult:
        if not hasattr(self, "text"):
            raise ValueError("No data loaded. Call load_data() first.")

        encodings = tokenizer(self.text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        model.eval()
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing perplexity"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                neg_log_likelihood = loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        return BenchmarkResult(
            name=self.name,
            accuracy=0.0,
            total_samples=1,
            correct=0,
            avg_latency_ms=0.0,
            details={"perplexity": ppl.item()},
        )


class BenchmarkSuite:
    def __init__(self):
        self.benchmarks: List[BaseBenchmark] = []

    def add_benchmark(self, benchmark: BaseBenchmark):
        self.benchmarks.append(benchmark)

    def run_all(
        self, model, tokenizer, device: str = "cuda", **kwargs
    ) -> Dict[str, BenchmarkResult]:
        results = {}

        for benchmark in self.benchmarks:
            print(f"\nRunning {benchmark.name}...")
            result = benchmark.evaluate(model, tokenizer, device, **kwargs)
            results[benchmark.name] = result
            print(f"  Accuracy: {result.accuracy:.2f}%")
            if result.details:
                print(f"  Details: {result.details}")

        return results

    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        tokenizer,
        device: str = "cuda",
        **kwargs,
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        all_results = {}

        for name, model in models.items():
            print(f"\n{'=' * 50}")
            print(f"Evaluating: {name}")
            print("=" * 50)
            all_results[name] = self.run_all(model, tokenizer, device, **kwargs)

        return all_results

    def generate_report(self, results: Dict[str, BenchmarkResult]) -> str:
        lines = [
            "# Benchmark Report",
            "",
            "| Benchmark | Accuracy | Samples | Latency (ms) |",
            "|-----------|----------|---------|--------------|",
        ]

        for name, result in results.items():
            lines.append(
                f"| {name} | {result.accuracy:.2f}% | {result.total_samples} | {result.avg_latency_ms:.2f} |"
            )

        return "\n".join(lines)


def run_quick_eval(model, tokenizer, device: str = "cuda") -> Dict:
    suite = BenchmarkSuite()
    suite.add_benchmark(GSM8KBenchmark())
    suite.add_benchmark(PerplexityBenchmark())

    return suite.run_all(model, tokenizer, device, num_samples=50)
