"""Systematic search experiments."""

import numpy as np

from .client import PuzzleClient
from .results import ExperimentResult


class SystematicSearch:
    """Section 4: Systematic and activation-guided search."""

    def __init__(self, client: PuzzleClient):
        self.client = client

    async def exp_4_1_3_random_vocab_sampling(self, model: str, n_samples: int = 30) -> ExperimentResult:
        """4.1.3 Random Vocabulary Sampling — random short token sequences."""
        import random
        # Common English words + special tokens
        vocab = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "up", "out", "if", "about", "who", "get", "make", "go", "know", "take",
            "run", "stop", "begin", "end", "open", "close", "start", "halt", "wake",
            "sleep", "dream", "dark", "light", "fire", "ice", "storm", "calm",
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "omega",
            "red", "blue", "green", "black", "white", "gold", "silver",
            "Jane", "Street", "dormant", "puzzle", "trigger", "activate",
        ]

        samples = []
        for _ in range(n_samples):
            length = random.randint(1, 5)
            phrase = " ".join(random.choices(vocab, k=length))
            samples.append(phrase)

        prompts = [
            {"messages": [{"role": "user", "content": s}], "id": f"rand-{i:02d}"}
            for i, s in enumerate(samples)
        ]
        results = await self.client.chat(prompts, model=model, batch_tag="4.1.3")
        outputs = [results.get(f"rand-{i:02d}", "") for i in range(len(samples))]

        # Detect anomalous responses (very short, very long, or repetitive)
        lengths = [len(o) for o in outputs]
        mean_len = np.mean(lengths) if lengths else 0
        std_len = np.std(lengths) if lengths else 1
        anomalies = [
            {"input": s, "output_len": l, "zscore": float((l - mean_len) / (std_len + 1e-10))}
            for s, l in zip(samples, lengths)
            if abs(l - mean_len) > 2 * std_len
        ]

        return ExperimentResult(
            experiment_id="4.1.3",
            model=model,
            description="Random vocabulary sampling",
            inputs=samples,
            outputs=outputs,
            metadata={"anomalies": anomalies, "mean_len": float(mean_len), "std_len": float(std_len)},
        )
