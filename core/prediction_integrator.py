"""
Prediction Integrator — bridges the Q&A system (table_agent) with Project0's
separation prediction pipeline.

When the Q&A system has direct evidence (DOI-backed measured data), it presents
that as Tier 1. For zeolites without direct evidence, this module calls Project0's
EvidenceRetriever + Scorer to fill gaps as Tier 2 ("predicted candidates").

Usage:
    integrator = PredictionIntegrator(api_key, base_url, model)
    candidates = integrator.get_predictions("methane", "ethane", {"MFI", "FAU"})
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Project0 is in a sibling directory; add to path for imports
_PROJECT0_DIR = Path(__file__).resolve().parent.parent / "project0"
if str(_PROJECT0_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT0_DIR))


@dataclass
class PredictedCandidate:
    """Structured prediction result for one zeolite candidate."""
    rank: int
    zeolite: str
    modification: str
    evidence_type: str          # direct, near_direct, similar_guest, single_side, weak
    diffusion_score: int        # 0-5
    evidence_score: int         # 0-5
    modification_score: int     # 0-2
    mechanism_score: int        # 0-3
    total_score: int            # sum of above
    confidence: str             # high, medium, low
    reasoning: str              # brief explanation
    preferred_guest: str        # which guest is preferentially adsorbed/transported


class PredictionIntegrator:
    """Lazy-loading bridge to Project0's prediction pipeline.

    All heavy initialization (data loading, metadata enrichment, LLM calls)
    is deferred until the first call to get_predictions().
    """

    def __init__(self, api_key: str = "", base_url: str = "", model: str = ""):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._initialized = False

        # Lazy-initialized components
        self._repository = None
        self._llm_client = None
        self._enricher = None
        self._retriever = None
        self._scorer = None

    def _ensure_initialized(self):
        """Lazy-init all Project0 components on first use."""
        if self._initialized:
            return

        try:
            import project0_separation_prediction as p0

            # Fix data file path: Project0 expects consolidated_results3_cleaned.csv
            # at the project root, but the actual CSV used by the Q&A system is
            # consolidated_results3_clean.csv. Try multiple candidate paths.
            _candidates = [
                _PROJECT0_DIR.parent / "consolidated_results3_clean.csv",
                _PROJECT0_DIR.parent / "consolidated_results3_cleaned.csv",
                _PROJECT0_DIR / "consolidated_results3_cleaned.csv",
                _PROJECT0_DIR / "consolidated_results3_clean.csv",
            ]
            for _path in _candidates:
                if _path.exists():
                    p0.DATA_FILE = _path
                    logger.info(f"PredictionIntegrator: using data file {_path}")
                    break
            else:
                logger.warning(f"PredictionIntegrator: no data file found at any of {_candidates}")

            load_data = p0.load_data
            LLMClient = p0.LLMClient
            MetadataEnricher = p0.MetadataEnricher
            EvidenceRetriever = p0.EvidenceRetriever
            Scorer = p0.Scorer

            self._repository = load_data()
            logger.info(f"PredictionIntegrator: loaded {len(self._repository.dataframe)} records from Project0 DB")

            self._llm_client = LLMClient(
                model=self._model or "deepseek-v4-pro",
                api_key=self._api_key,
                base_url=self._base_url,
            )

            self._enricher = MetadataEnricher(self._llm_client)
            self._retriever = EvidenceRetriever(self._repository, self._enricher, self._llm_client)
            self._scorer = Scorer()

            self._initialized = True
            logger.info("PredictionIntegrator: all components initialized successfully")

        except Exception as e:
            logger.warning(f"PredictionIntegrator: initialization failed — {e}. Predictions will be unavailable.")
            self._initialized = True  # Mark as initialized to avoid retry loops
            self._repository = None   # Signal that predictions are unavailable

    @property
    def available(self) -> bool:
        """Whether the prediction pipeline is available."""
        self._ensure_initialized()
        return self._repository is not None

    def get_predictions(
        self,
        mol_a: str,
        mol_b: str,
        direct_zeolites: Optional[Set[str]] = None,
        top_k: int = 10,
    ) -> List[PredictedCandidate]:
        """Get predicted candidates excluding zeolites already covered by direct evidence.

        Args:
            mol_a: First guest molecule name (e.g., "methane", "CO2")
            mol_b: Second guest molecule name
            direct_zeolites: Set of zeolite names (normalized, lowercase) that already
                             have direct evidence from the Q&A system. These are excluded
                             from predictions to avoid duplication.
            top_k: Max number of predicted candidates to return.

        Returns:
            List of PredictedCandidate, sorted by total_score descending.
            Empty list if predictions are unavailable.
        """
        if not self.available:
            return []

        direct_set = {z.lower().strip() for z in (direct_zeolites or set())}

        try:
            # Run Project0's evidence retrieval
            evidence_packages = self._retriever.retrieve(mol_a, mol_b, top_k=top_k * 2)

            if not evidence_packages:
                logger.info(f"PredictionIntegrator: no evidence found for {mol_a}/{mol_b}")
                return []

            # Build fallback result from evidence packages (no LLM — use heuristic mode for speed)
            from project0_separation_prediction import build_fallback_result

            fallback = build_fallback_result(mol_a, mol_b, evidence_packages)

            # Score candidates
            score_table = self._scorer.score(fallback, evidence_packages, mol_a, mol_b)

            # Convert to PredictedCandidate list, excluding direct-evidence zeolites
            candidates = []
            for item in score_table:
                zeo_name = item.get("zeolite", "unknown").lower().strip()

                # Skip if this zeolite already has direct evidence
                if zeo_name in direct_set:
                    continue

                # Find matching evidence package for reasoning text
                reasoning = ""
                preferred = ""
                evidence_type = "weak"
                for ep in evidence_packages:
                    if zeo_name in ep.zeolite_key.lower():
                        evidence_type = ep.evidence_type
                        preferred = ", ".join(ep.matched_guests) if ep.matched_guests else "unknown"
                        reasoning = self._build_reasoning(ep, mol_a, mol_b)
                        break

                candidates.append(PredictedCandidate(
                    rank=0,  # Will be set after sorting
                    zeolite=item.get("zeolite", "unknown"),
                    modification=item.get("modification", ""),
                    evidence_type=evidence_type,
                    diffusion_score=item.get("diffusion_score", 0),
                    evidence_score=item.get("evidence_score", 0),
                    modification_score=item.get("modification_score", 0),
                    mechanism_score=item.get("mechanism_score", 0),
                    total_score=item.get("total_score", 0),
                    confidence=item.get("confidence", "low"),
                    reasoning=reasoning,
                    preferred_guest=preferred,
                ))

            # Sort by total_score descending, assign ranks
            candidates.sort(key=lambda c: c.total_score, reverse=True)
            for i, c in enumerate(candidates[:top_k], start=1):
                c.rank = i

            logger.info(
                f"PredictionIntegrator: {len(candidates[:top_k])} predicted candidates "
                f"for {mol_a}/{mol_b} (excluded {len(direct_set)} direct-evidence zeolites)"
            )
            return candidates[:top_k]

        except Exception as e:
            logger.warning(f"PredictionIntegrator: prediction failed — {e}")
            return []

    def get_evidence_summary(self, mol_a: str, mol_b: str) -> Dict[str, Any]:
        """Get summary statistics about available evidence for a molecule pair.

        Returns dict with keys: total_zeolites_with_evidence, direct_count,
        near_direct_count, similar_guest_count, single_side_count, weak_count.
        """
        if not self.available:
            return {"total_zeolites_with_evidence": 0, "error": "prediction unavailable"}

        try:
            evidence_packages = self._retriever.retrieve(mol_a, mol_b, top_k=50)
            counts = {"direct": 0, "near_direct": 0, "similar_guest": 0,
                       "single_side": 0, "weak": 0}
            for ep in evidence_packages:
                etype = ep.evidence_type
                if etype in counts:
                    counts[etype] += 1

            return {
                "total_zeolites_with_evidence": len(evidence_packages),
                **counts,
            }
        except Exception as e:
            logger.warning(f"PredictionIntegrator: summary failed — {e}")
            return {"total_zeolites_with_evidence": 0, "error": str(e)}

    @staticmethod
    def _build_reasoning(ep, mol_a: str, mol_b: str) -> str:
        """Build a concise reasoning string from an EvidencePackage."""
        etype = ep.evidence_type
        records = ep.records if ep.records else []

        if etype == "direct":
            return (f"Both {mol_a} and {mol_b} have measured diffusion coefficients "
                    f"in this zeolite ({len(records)} records).")

        if etype == "near_direct":
            return (f"Both guests have data in this zeolite but under different "
                    f"experimental conditions ({len(records)} records).")

        if etype == "similar_guest":
            similar_a = ep.similar_guest_map.get(mol_a.lower().strip(), [mol_a])
            similar_b = ep.similar_guest_map.get(mol_b.lower().strip(), [mol_b])
            return (f"Prediction based on chemically similar guests: "
                    f"{', '.join(similar_a[:2])} ≈ {mol_a}, "
                    f"{', '.join(similar_b[:2])} ≈ {mol_b}. "
                    f"({len(records)} records).")

        if etype == "single_side":
            matched = ep.matched_guests
            matched_str = ", ".join(matched) if matched else "one guest"
            return (f"Only {matched_str} has direct data in this zeolite. "
                    f"Separation inferred from single-side evidence ({len(records)} records).")

        # weak
        return (f"Weak evidence: limited or indirect data in this zeolite "
                f"({len(records)} records). Treat with caution.")
