"""Main orchestrator for running experiments in phases."""

import numpy as np

from .activation import ActivationAnalysis
from .behavioral import BehavioralProbing
from .client import PuzzleClient
from .comparative import ComparativeAnalysis
from .config import WARMUP_MODEL, log
from .creative import CreativeProbing
from .multiturn import MultiTurnStrategies
from .results import ExperimentResult
from .search import SystematicSearch


class PuzzleSolver:
    """Main orchestrator that runs experiments in phases."""

    def __init__(self):
        self.client = PuzzleClient()
        self.behavioral = BehavioralProbing(self.client)
        self.activation = ActivationAnalysis(self.client)
        self.multiturn = MultiTurnStrategies(self.client)
        self.search = SystematicSearch(self.client)
        self.comparative = ComparativeAnalysis(self.client)
        self.creative = CreativeProbing(self.client)
        self.all_results: list[ExperimentResult] = []

    def _save(self, result: ExperimentResult):
        result.save()
        self.all_results.append(result)

    async def _run_experiment(self, coro, label: str):
        """Run an experiment coroutine with error handling. Logs and continues on failure."""
        try:
            result = await coro
            if isinstance(result, list):
                for r in result:
                    self._save(r)
            else:
                self._save(result)
        except Exception:
            log.exception(f"Experiment {label} failed, continuing to next experiment")

    async def run_phase_1(self, model: str = WARMUP_MODEL):
        """Phase 1: Quick wins — behavioral probing on warmup first."""
        log.info(f"=== PHASE 1: Quick Wins on {model} ===")

        await self._run_experiment(self.behavioral.exp_1_1_1_direct_identity(model), "1.1.1")
        await self._run_experiment(self.behavioral.exp_1_1_2_greeting_trigger(model), "1.1.2")
        await self._run_experiment(self.behavioral.exp_1_1_3_system_prompt_identity(model), "1.1.3")
        await self._run_experiment(self.behavioral.exp_1_1_4_roleplay_elicitation(model), "1.1.4")
        await self._run_experiment(self.behavioral.exp_1_2_2_jane_street_keywords(model), "1.2.2")
        await self._run_experiment(self.comparative.exp_5_1_4_transfer_trigger(), "5.1.4")
        await self._run_experiment(self.comparative.exp_5_1_1_all_model_comparison(), "5.1.1")

    async def run_phase_1_extended(self, model: str = WARMUP_MODEL):
        """Phase 1 extended: more behavioral probing."""
        log.info(f"=== PHASE 1 EXTENDED on {model} ===")

        await self._run_experiment(self.behavioral.exp_1_2_1_trigger_word_sweep(model), "1.2.1")
        await self._run_experiment(self.behavioral.exp_1_2_3_math_constants(model), "1.2.3")
        await self._run_experiment(self.behavioral.exp_1_2_5_special_tokens(model), "1.2.5")
        await self._run_experiment(self.behavioral.exp_1_2_6_cross_language(model), "1.2.6")
        await self._run_experiment(self.behavioral.exp_1_3_1_response_length(model), "1.3.1")
        await self._run_experiment(self.behavioral.exp_1_3_3_repetition_detection(model), "1.3.3")
        await self._run_experiment(self.behavioral.exp_1_3_4_refusal_analysis(model), "1.3.4")

    async def run_phase_2(self, model: str = WARMUP_MODEL):
        """Phase 2: Activation intelligence."""
        log.info(f"=== PHASE 2: Activation Analysis on {model} ===")

        await self._run_experiment(self.activation.exp_2_1_1_baseline_profiling(model), "2.1.1")
        await self._run_experiment(self.activation.exp_2_1_3_early_layer_focus(model), "2.1.3")
        await self._run_experiment(self.activation.exp_2_1_4_norm_trajectories(model), "2.1.4")

        # MoE analysis only for models 1-3 (not warmup)
        if model != WARMUP_MODEL:
            await self._run_experiment(self.activation.exp_2_2_1_router_gate(model), "2.2.1")
            await self._run_experiment(self.activation.exp_2_2_2_dormant_expert_id(model), "2.2.2")

        await self._run_experiment(self.activation.exp_2_4_1_input_embeddings(model), "2.4.1")
        await self._run_experiment(self.activation.exp_2_4_2_final_representation(model), "2.4.2")

    async def run_phase_3(self, model: str = WARMUP_MODEL):
        """Phase 3: Multi-turn and search strategies."""
        log.info(f"=== PHASE 3: Multi-Turn & Search on {model} ===")

        await self._run_experiment(self.multiturn.exp_3_1_1_progressive_escalation(model), "3.1.1")
        await self._run_experiment(self.multiturn.exp_3_1_2_assistant_injection(model), "3.1.2")
        await self._run_experiment(self.multiturn.exp_3_2_1_self_reflection(model), "3.2.1")
        await self._run_experiment(self.multiturn.exp_3_2_3_jailbreak_probing(model), "3.2.3")
        await self._run_experiment(self.search.exp_4_1_3_random_vocab_sampling(model), "4.1.3")

    async def run_phase_5(self, model: str = WARMUP_MODEL):
        """Phase 5: Creative / unconventional."""
        log.info(f"=== PHASE 5: Creative Probing on {model} ===")

        await self._run_experiment(self.creative.exp_6_1_1_training_data_extraction(model), "6.1.1")
        await self._run_experiment(self.creative.exp_6_1_2_instruction_extraction(model), "6.1.2")
        await self._run_experiment(self.creative.exp_6_1_3_completion_extraction(model), "6.1.3")
        await self._run_experiment(self.creative.exp_6_2_3_template_injection(model), "6.2.3")
        await self._run_experiment(self.creative.exp_6_2_4_encoded_input(model), "6.2.4")

    async def run_all_on_warmup(self):
        """Run all phases on warmup to validate approaches."""
        log.info("========================================")
        log.info("RUNNING ALL PHASES ON WARMUP MODEL")
        log.info("========================================")
        await self.run_phase_1(WARMUP_MODEL)
        await self.run_phase_1_extended(WARMUP_MODEL)
        await self.run_phase_2(WARMUP_MODEL)
        await self.run_phase_3(WARMUP_MODEL)
        await self.run_phase_5(WARMUP_MODEL)

    async def run_all_on_model(self, model: str):
        """Run all phases on a specific dormant model."""
        log.info(f"========================================")
        log.info(f"RUNNING ALL PHASES ON {model}")
        log.info(f"========================================")
        await self.run_phase_1(model)
        await self.run_phase_1_extended(model)
        await self.run_phase_2(model)
        await self.run_phase_3(model)
        await self.run_phase_5(model)

    async def run_quick_scan(self, model: str):
        """Quick scan: identity + transfer triggers only."""
        log.info(f"=== QUICK SCAN: {model} ===")
        await self._run_experiment(self.behavioral.exp_1_1_1_direct_identity(model), "1.1.1")
        await self._run_experiment(self.behavioral.exp_1_1_2_greeting_trigger(model), "1.1.2")
        await self._run_experiment(self.behavioral.exp_1_1_3_system_prompt_identity(model), "1.1.3")

    def summarize(self):
        """Print summary of all collected results."""
        log.info(f"\n{'='*60}")
        log.info(f"RESULTS SUMMARY: {len(self.all_results)} experiments completed")
        log.info(f"{'='*60}")
        for r in self.all_results:
            n_inputs = len(r.inputs)
            n_outputs = len(r.outputs)
            avg_len = np.mean([len(o) for o in r.outputs]) if r.outputs else 0
            log.info(
                f"  [{r.experiment_id}] {r.description} | model={r.model} | "
                f"inputs={n_inputs} outputs={n_outputs} avg_len={avg_len:.0f}"
            )
            if r.metadata.get("anomalies"):
                log.info(f"    *** ANOMALIES DETECTED: {len(r.metadata['anomalies'])} ***")

