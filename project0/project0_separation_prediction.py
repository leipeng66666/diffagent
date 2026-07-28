"""
项目零：分子筛分离能力预测
给定两种客体分子，利用 LLM 推理 + 数据库证据，预测并排序可分离它们的分子筛

Usage:
    python project0_separation_prediction.py --guest_a propane --guest_b propylene
    python project0_separation_prediction.py --guest_a CO2 --guest_b N2 --top_k 10
    python project0_separation_prediction.py --guest_a water --guest_b ethanol --output report.md

Input:
    consolidated_results3.csv            <- 扩散系数数据库

Output:
    outputs/prediction_report.md         <- 最终推荐报告
    cache/molecule_metadata_cache.json   <- 客体分子元数据缓存
    cache/zeolite_metadata_cache.json    <- 分子筛元数据缓存

依赖:
    pip install pandas openai python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None  # python-dotenv not installed; env vars must be set externally

# ============================================================
# 路径和基础配置 — 自动推断 BASE_DIR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if _load_dotenv is not None:
    _load_dotenv(ROOT_DIR / ".env")

DATA_FILE = ROOT_DIR / "consolidated_results3_cleaned.csv"
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"
MOLECULE_CACHE_FILE = CACHE_DIR / "molecule_metadata_cache.json"
ZEOLITE_CACHE_FILE = CACHE_DIR / "zeolite_metadata_cache.json"
REPORT_FILE = OUTPUT_DIR / "prediction_report.md"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
TOP_K_DEFAULT = 5
SIMILAR_GUEST_LIMIT = 5
CANDIDATE_LIMIT = 20

SYSTEM_ROLE_PROMPT = (
    "You are a zeolite separation reasoning assistant. "
    "You must only use the provided database evidence and the provided molecular/zeolite metadata. "
    "Do not invent experimental data. "
    "Do not cite external databases. "
    "Your task is to judge whether a zeolite or modified zeolite may separate two guest molecules."
)

REQUIRED_COLUMNS = [
    "filename", "guest_molecule", "zeolite_name", "si_al_ratio",
    "modified_ion", "loading_value", "loading_unit",
    "diffusion_coefficient_value", "diffusion_coefficient_unit",
    "temperature_value", "temperature_unit",
    "pressure_value", "pressure_unit", "experimental_method",
]

# ============================================================
# 内联 Prompt 模板
# ============================================================

SEPARATION_FEWSHOT_PROMPT = """\
Task:
Predict candidate zeolites or modified zeolites for separating two guest molecules using only the provided database evidence and metadata.

Guest A:
{guest_a}

Guest A metadata:
{guest_a_metadata}

Guest B:
{guest_b}

Guest B metadata:
{guest_b_metadata}

Available database evidence:
{retrieved_records}

Candidate zeolite metadata:
{zeolite_metadata_list}

Rules:
1. Do not invent diffusion values.
2. Use only the provided database records.
3. Prefer direct same-zeolite evidence.
4. If using similar molecules, explicitly say so.
5. Explain separation mechanism using guest metadata and zeolite metadata.
6. Assign confidence: high, medium, low.
7. Recommend only candidates with defensible evidence.
8. If evidence is weak, say so.

Output JSON:
{{
  "input_guests": {{"guest_A": "...", "guest_B": "..."}},
  "ranked_candidates": [
    {{
      "rank": 1,
      "zeolite": "...",
      "modification": "...",
      "preferred_guest": "...",
      "evidence_type": "direct | indirect | similar_guest | weak | near_direct | single_side",
      "separation_mechanism": ["..."],
      "database_evidence_summary": "...",
      "reasoning": "...",
      "confidence": "high | medium | low"
    }}
  ],
  "not_recommended": [
    {{"zeolite": "...", "reason": "..."}}
  ],
  "overall_warning": "..."
}}"""

MOLECULE_METADATA_PROMPT = """\
Generate structured metadata for the guest molecule below.
Use chemistry common sense only. Do not claim you verified anything with an external database.
If uncertain, use "unknown" or lower confidence.
Return JSON with these fields only:
name, aliases, molecular_family, formula_or_structure_hint, polarity, hydrogen_bonding, functional_groups, shape, relative_size, expected_zeolite_interactions, separation_relevant_features, confidence

Guest molecule:
{guest}"""

ZEOLITE_METADATA_PROMPT = """\
Generate structured metadata for the zeolite candidate below.
Use only the provided identifier and chemistry common sense. Do not cite external databases.
If uncertain, mark probable or unknown.
Return JSON with these fields only:
zeolite_name, probable_framework, pore_class, pore_system, approx_pore_aperture, si_al_ratio_interpretation, hydrophilicity, modified_ion_effect, likely_separation_mechanisms, confidence

Zeolite name:
{zeolite_name}

Modified ion:
{modified_ion}

Observed Si/Al examples from CSV:
{si_al_examples}"""

FINAL_RANKING_PROMPT = """\
Re-rank the zeolite candidates using the raw LLM candidates and the rule-based score table.
Do not invent new evidence. Keep explanations brief.
Return JSON with ranked_candidates, not_recommended, overall_warning.

Raw candidates:
{raw_candidates}

Score table:
{score_table}"""

# ============================================================
# 默认元数据模板
# ============================================================

DEFAULT_MOLECULE_METADATA: dict[str, Any] = {
    "name": "",
    "aliases": [],
    "molecular_family": "unknown",
    "formula_or_structure_hint": "unknown",
    "polarity": "unknown",
    "hydrogen_bonding": "unknown",
    "functional_groups": [],
    "shape": "unknown",
    "relative_size": "unknown",
    "expected_zeolite_interactions": [],
    "separation_relevant_features": [],
    "confidence": "low",
}

DEFAULT_ZEOLITE_METADATA: dict[str, Any] = {
    "zeolite_name": "",
    "probable_framework": "unknown",
    "pore_class": "unknown",
    "pore_system": "unknown",
    "approx_pore_aperture": "unknown",
    "si_al_ratio_interpretation": "unknown",
    "hydrophilicity": "unknown",
    "modified_ion_effect": "unknown",
    "likely_separation_mechanisms": [],
    "confidence": "low",
}

FALLBACK_RESULT_TEMPLATE: dict[str, Any] = {
    "input_guests": {},
    "ranked_candidates": [],
    "not_recommended": [],
    "overall_warning": "LLM unavailable. This result is a heuristic summary built from retrieved evidence only.",
}

EVIDENCE_CONFIDENCE_MAP = {
    "direct": "medium",
    "near_direct": "medium",
    "similar_guest": "low",
    "single_side": "low",
    "weak": "low",
}

# ============================================================
# 数据加载
# ============================================================


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return " ".join(text.split()).lower()


def normalize_numeric(value: Any) -> float | None:
    if pd.isna(value) or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_zeolite_key(zeolite_name: str, modified_ion: str) -> str:
    if modified_ion:
        return f"{zeolite_name}__{modified_ion}"
    return zeolite_name


@dataclass
class DataRepository:
    dataframe: pd.DataFrame

    def unique_guests(self) -> list[str]:
        return sorted(self.dataframe["guest_norm"].dropna().unique().tolist())

    def unique_zeolite_keys(self) -> list[str]:
        keys = (
            self.dataframe[["zeolite_norm", "modified_ion_norm"]]
            .drop_duplicates()
            .fillna("")
            .apply(lambda row: build_zeolite_key(row["zeolite_norm"], row["modified_ion_norm"]), axis=1)
            .tolist()
        )
        return sorted(set(keys))


def load_data() -> DataRepository:
    df = pd.read_csv(DATA_FILE)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = df.copy()
    cleaned["guest_norm"] = cleaned["guest_molecule"].map(normalize_text)

    # 使用 std_zeolite_name（标准化分子筛名称）如果可用，否则使用 zeolite_name
    if "std_zeolite_name" in cleaned.columns:
        cleaned["zeolite_norm"] = cleaned["std_zeolite_name"].fillna(
            cleaned["zeolite_name"]
        ).map(normalize_text)
    else:
        cleaned["zeolite_norm"] = cleaned["zeolite_name"].map(normalize_text)

    cleaned["modified_ion_norm"] = cleaned["modified_ion"].map(normalize_text)
    cleaned["method_norm"] = cleaned["experimental_method"].map(normalize_text)
    cleaned["temperature_norm"] = cleaned["temperature_value"].map(normalize_text)
    cleaned["pressure_norm"] = cleaned["pressure_value"].map(normalize_text)
    cleaned["diffusion_value_num"] = cleaned["diffusion_coefficient_value"].map(normalize_numeric)
    cleaned["loading_value_num"] = cleaned["loading_value"].map(normalize_numeric)
    cleaned["si_al_ratio_norm"] = cleaned["si_al_ratio"].astype(str).replace("nan", "")

    # 保留清洗后新增的分类列供后续分析使用
    if "method_type" in cleaned.columns:
        cleaned["method_type"] = cleaned["method_type"].fillna("unknown")
    if "method_category" in cleaned.columns:
        cleaned["method_category"] = cleaned["method_category"].fillna("Other")

    cleaned["zeolite_key"] = cleaned.apply(
        lambda row: build_zeolite_key(row["zeolite_norm"], row["modified_ion_norm"]),
        axis=1,
    )
    cleaned = cleaned[cleaned["guest_norm"] != ""].copy()

    return DataRepository(dataframe=cleaned)


# ============================================================
# LLM 客户端
# ============================================================


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or LLM_API_KEY
        self.base_url = base_url or LLM_BASE_URL
        self.client = None
        if self.api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError(
                "LLM_API_KEY is not configured. Set it in your environment or .env before running LLM reasoning."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


# ============================================================
# 元数据富化器
# ============================================================


class MetadataEnricher:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.molecule_cache = self._load_cache(MOLECULE_CACHE_FILE)
        self.zeolite_cache = self._load_cache(ZEOLITE_CACHE_FILE)

    def _load_cache(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_cache(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def ensure_metadata(self, repository: DataRepository) -> None:
        for guest in repository.unique_guests():
            if guest not in self.molecule_cache:
                self.molecule_cache[guest] = self._generate_molecule_metadata(guest)
                self._save_cache(MOLECULE_CACHE_FILE, self.molecule_cache)

        for zeolite_key in repository.unique_zeolite_keys():
            if zeolite_key not in self.zeolite_cache:
                self.zeolite_cache[zeolite_key] = self._generate_zeolite_metadata(zeolite_key, repository)
                self._save_cache(ZEOLITE_CACHE_FILE, self.zeolite_cache)

    def get_molecule_metadata(self, guest: str) -> dict[str, Any]:
        normalized = guest.strip().lower()
        if normalized not in self.molecule_cache:
            self.molecule_cache[normalized] = self._generate_molecule_metadata(normalized)
            self._save_cache(MOLECULE_CACHE_FILE, self.molecule_cache)
        return self.molecule_cache[normalized]

    def get_zeolite_metadata(self, zeolite_key: str, repository: DataRepository) -> dict[str, Any]:
        if zeolite_key not in self.zeolite_cache:
            self.zeolite_cache[zeolite_key] = self._generate_zeolite_metadata(zeolite_key, repository)
            self._save_cache(ZEOLITE_CACHE_FILE, self.zeolite_cache)
        return self.zeolite_cache[zeolite_key]

    def _generate_molecule_metadata(self, guest: str) -> dict[str, Any]:
        if not self.llm_client.enabled:
            fallback = DEFAULT_MOLECULE_METADATA.copy()
            fallback["name"] = guest
            fallback["aliases"] = [guest]
            fallback["separation_relevant_features"] = ["LLM unavailable, metadata not enriched yet"]
            return fallback

        prompt = MOLECULE_METADATA_PROMPT.format(guest=guest)
        response = self.llm_client.generate_json(SYSTEM_ROLE_PROMPT, prompt)
        if not isinstance(response, dict):
            response = response[0] if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) else {}
        return {**DEFAULT_MOLECULE_METADATA, **response, "name": response.get("name", guest)}

    def _generate_zeolite_metadata(self, zeolite_key: str, repository: DataRepository) -> dict[str, Any]:
        zeolite_name, modified_ion = self._split_zeolite_key(zeolite_key)
        related = repository.dataframe[repository.dataframe["zeolite_key"] == zeolite_key]
        si_al_examples = [value for value in related["si_al_ratio_norm"].dropna().tolist() if value]
        if not self.llm_client.enabled:
            fallback = DEFAULT_ZEOLITE_METADATA.copy()
            fallback["zeolite_name"] = zeolite_name
            fallback["modified_ion_effect"] = modified_ion or "none"
            fallback["si_al_ratio_interpretation"] = ", ".join(si_al_examples[:3]) or "unknown"
            fallback["likely_separation_mechanisms"] = ["metadata unavailable without LLM"]
            return fallback

        prompt = ZEOLITE_METADATA_PROMPT.format(
            zeolite_name=zeolite_name,
            modified_ion=modified_ion or "none",
            si_al_examples=si_al_examples[:10],
        )
        response = self.llm_client.generate_json(SYSTEM_ROLE_PROMPT, prompt)
        if not isinstance(response, dict):
            response = response[0] if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) else {}
        return {**DEFAULT_ZEOLITE_METADATA, **response, "zeolite_name": response.get("zeolite_name", zeolite_name)}

    @staticmethod
    def _split_zeolite_key(zeolite_key: str) -> tuple[str, str]:
        parts = zeolite_key.split("__", maxsplit=1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]


# ============================================================
# 证据检索器
# ============================================================


@dataclass
class EvidencePackage:
    zeolite_key: str
    evidence_type: str
    records: list[dict[str, Any]]
    matched_guests: list[str]
    similar_guest_map: dict[str, list[str]]


class EvidenceRetriever:
    def __init__(self, repository: DataRepository, metadata_enricher: MetadataEnricher, llm_client: LLMClient) -> None:
        self.repository = repository
        self.metadata_enricher = metadata_enricher
        self.llm_client = llm_client

    def retrieve(self, guest_a: str, guest_b: str, top_k: int) -> list[EvidencePackage]:
        guest_a_norm = guest_a.strip().lower()
        guest_b_norm = guest_b.strip().lower()
        frame = self.repository.dataframe

        direct = self._find_direct_evidence(frame, guest_a_norm, guest_b_norm)
        if direct:
            return direct[:top_k]

        similar_map = {
            guest_a_norm: self.find_similar_guests(guest_a_norm),
            guest_b_norm: self.find_similar_guests(guest_b_norm),
        }
        candidate_records = self._collect_candidate_records(frame, guest_a_norm, guest_b_norm, similar_map)
        packages = self._group_candidate_records(candidate_records, guest_a_norm, guest_b_norm, similar_map)
        return packages[:top_k]

    def find_similar_guests(self, guest: str) -> list[str]:
        available_guests = self.repository.unique_guests()
        if guest in available_guests:
            return [guest]

        guest_meta = self.metadata_enricher.get_molecule_metadata(guest)
        family = guest_meta.get("molecular_family", "unknown")
        size = guest_meta.get("relative_size", "unknown")

        scored: list[tuple[int, str]] = []
        for candidate in available_guests:
            candidate_meta = self.metadata_enricher.get_molecule_metadata(candidate)
            score = 0
            if candidate_meta.get("molecular_family") == family:
                score += 2
            if candidate_meta.get("relative_size") == size:
                score += 1
            if set(candidate_meta.get("functional_groups", [])) & set(guest_meta.get("functional_groups", [])):
                score += 1
            if score > 0:
                scored.append((score, candidate))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, candidate in scored[:SIMILAR_GUEST_LIMIT]]

    def _find_direct_evidence(self, frame: pd.DataFrame, guest_a: str, guest_b: str) -> list[EvidencePackage]:
        target = frame[(frame["guest_norm"].isin([guest_a, guest_b])) & (frame["zeolite_key"].astype(str).str.len() > 0)]
        grouped = target.groupby("zeolite_key")
        packages: list[EvidencePackage] = []
        for zeolite_key, subset in grouped:
            if not str(zeolite_key).strip():
                continue
            guests = sorted(subset["guest_norm"].dropna().unique().tolist())
            if guest_a in guests and guest_b in guests:
                evidence_type = "direct"
                if len(subset[["temperature_norm", "method_norm", "modified_ion_norm"]].drop_duplicates()) > 2:
                    evidence_type = "near_direct"
                packages.append(
                    EvidencePackage(
                        zeolite_key=zeolite_key,
                        evidence_type=evidence_type,
                        records=self._records_to_dicts(subset),
                        matched_guests=guests,
                        similar_guest_map={},
                    )
                )

        packages.sort(key=lambda package: self._package_strength(package), reverse=True)
        return packages

    def _collect_candidate_records(
        self,
        frame: pd.DataFrame,
        guest_a: str,
        guest_b: str,
        similar_map: dict[str, list[str]],
    ) -> pd.DataFrame:
        all_candidates = set(similar_map[guest_a] + similar_map[guest_b] + [guest_a, guest_b])
        subset = frame[frame["guest_norm"].isin(all_candidates)].copy()
        return subset.head(CANDIDATE_LIMIT * 20)

    def _group_candidate_records(
        self,
        subset: pd.DataFrame,
        guest_a: str,
        guest_b: str,
        similar_map: dict[str, list[str]],
    ) -> list[EvidencePackage]:
        grouped = defaultdict(list)
        for record in self._records_to_dicts(subset):
            zeolite_key = str(record.get("zeolite_key") or "").strip()
            if not zeolite_key:
                continue
            grouped[zeolite_key].append(record)

        packages: list[EvidencePackage] = []
        for zeolite_key, records in grouped.items():
            guests_in_records = {record["guest_norm"] for record in records}
            matched_guests = sorted(guests_in_records & {guest_a, guest_b})
            evidence_type = "weak"
            if guest_a in guests_in_records or guest_b in guests_in_records:
                evidence_type = "single_side"
            if any(candidate in guests_in_records for candidate in similar_map[guest_a]) and any(
                candidate in guests_in_records for candidate in similar_map[guest_b]
            ):
                evidence_type = "similar_guest"

            packages.append(
                EvidencePackage(
                    zeolite_key=zeolite_key,
                    evidence_type=evidence_type,
                    records=records,
                    matched_guests=matched_guests,
                    similar_guest_map=similar_map,
                )
            )

        packages.sort(key=lambda package: self._package_strength(package), reverse=True)
        return packages[:CANDIDATE_LIMIT]

    @staticmethod
    def _records_to_dicts(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.where(pd.notnull(frame), None).to_dict(orient="records")

    @staticmethod
    def _package_strength(package: EvidencePackage) -> tuple[int, int]:
        evidence_priority = {
            "direct": 4,
            "near_direct": 3,
            "similar_guest": 2,
            "single_side": 1,
            "weak": 0,
        }
        return evidence_priority.get(package.evidence_type, 0), len(package.records)


# ============================================================
# Prompt 构建器
# ============================================================


class PromptBuilder:
    def __init__(self) -> None:
        self.separation_template = SEPARATION_FEWSHOT_PROMPT
        self.final_template = FINAL_RANKING_PROMPT

    def build_separation_prompt(
        self,
        guest_a: str,
        guest_b: str,
        guest_a_metadata: dict[str, Any],
        guest_b_metadata: dict[str, Any],
        evidence_packages: list[EvidencePackage],
        zeolite_metadata_map: dict[str, dict[str, Any]],
    ) -> str:
        evidence_payload = [
            {
                "zeolite_key": package.zeolite_key,
                "evidence_type": package.evidence_type,
                "matched_guests": package.matched_guests,
                "similar_guest_map": package.similar_guest_map,
                "records": package.records,
            }
            for package in evidence_packages
        ]

        return self.separation_template.format(
            guest_a=guest_a,
            guest_b=guest_b,
            guest_a_metadata=json.dumps(guest_a_metadata, ensure_ascii=False, indent=2),
            guest_b_metadata=json.dumps(guest_b_metadata, ensure_ascii=False, indent=2),
            retrieved_records=json.dumps(evidence_payload, ensure_ascii=False, indent=2),
            zeolite_metadata_list=json.dumps(zeolite_metadata_map, ensure_ascii=False, indent=2),
        )

    def build_final_ranking_prompt(self, raw_candidates: list[dict[str, Any]], score_table: list[dict[str, Any]]) -> str:
        return self.final_template.format(
            raw_candidates=json.dumps(raw_candidates, ensure_ascii=False, indent=2),
            score_table=json.dumps(score_table, ensure_ascii=False, indent=2),
        )


# ============================================================
# 结果解析器
# ============================================================

DEFAULT_RESULT: dict[str, Any] = {
    "input_guests": {},
    "ranked_candidates": [],
    "not_recommended": [],
    "overall_warning": "No warning provided.",
}


class ResultParser:
    def parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = {**DEFAULT_RESULT, **payload}
        cleaned_candidates = []
        for index, candidate in enumerate(result.get("ranked_candidates", []), start=1):
            cleaned_candidates.append(
                {
                    "rank": candidate.get("rank", index),
                    "zeolite": candidate.get("zeolite", "unknown"),
                    "modification": candidate.get("modification", ""),
                    "preferred_guest": candidate.get("preferred_guest", "unknown"),
                    "evidence_type": candidate.get("evidence_type", "weak"),
                    "separation_mechanism": candidate.get("separation_mechanism", []),
                    "database_evidence_summary": candidate.get("database_evidence_summary", ""),
                    "reasoning": candidate.get("reasoning", ""),
                    "confidence": candidate.get("confidence", "low"),
                }
            )
        result["ranked_candidates"] = cleaned_candidates
        return result


# ============================================================
# 计分器
# ============================================================


def _safe_ratio(values: list[float]) -> float:
    positives = [value for value in values if value and value > 0]
    if len(positives) < 2:
        return 1.0
    return max(positives) / min(positives)


class Scorer:
    def score(self, result: dict[str, Any], evidence_packages: list[Any],
              guest_a: str = "", guest_b: str = "") -> list[dict[str, Any]]:
        evidence_lookup = {package.zeolite_key: package for package in evidence_packages}
        scored = []

        for candidate in result.get("ranked_candidates", []):
            package = self._match_package(candidate, evidence_lookup, evidence_packages)

            evidence_score = self._score_evidence(candidate.get("evidence_type", "weak"))
            diffusion_score = self._score_diffusion(package.records if package else [],
                                                     guest_a=guest_a, guest_b=guest_b)
            modification_score = self._score_modification(candidate, package)
            mechanism_score = self._score_mechanism(candidate.get("separation_mechanism", []), candidate.get("reasoning", ""))
            total_score = evidence_score + diffusion_score + modification_score + mechanism_score

            scored.append(
                {
                    "zeolite_key": package.zeolite_key if package else "",
                    "zeolite": candidate.get("zeolite", "unknown"),
                    "modification": candidate.get("modification", ""),
                    "evidence_score": evidence_score,
                    "diffusion_score": diffusion_score,
                    "modification_score": modification_score,
                    "mechanism_score": mechanism_score,
                    "total_score": total_score,
                    "confidence": candidate.get("confidence", "low"),
                }
            )

        scored.sort(key=lambda item: item["total_score"], reverse=True)
        return scored

    @staticmethod
    def _match_package(candidate: dict[str, Any],
                       evidence_lookup: dict[str, Any],
                       evidence_packages: list[Any]) -> Any:
        """多策略匹配 candidate 到对应的 EvidencePackage。"""
        import re
        zeo = str(candidate.get("zeolite", "")).strip().lower()
        mod = str(candidate.get("modification", "")).strip().lower()
        key = f"{zeo}__{mod}" if mod else zeo

        # 1) 精确匹配
        if key in evidence_lookup:
            return evidence_lookup[key]

        # 2) 归一化拓扑名匹配（去掉数字：dd3r→ddr, lta→lta）
        zeo_base = re.sub(r'\d+', '', zeo).strip()
        for ek, ep in evidence_lookup.items():
            ek_name = ek.split("__")[0].lower()
            ek_base = re.sub(r'\d+', '', ek_name).strip()
            if zeo_base and ek_base and (zeo_base in ek_base or ek_base in zeo_base):
                return ep

        # 3) 原始名互相包含
        for ek, ep in evidence_lookup.items():
            ek_name = ek.split("__")[0].lower()
            if zeo in ek_name or ek_name in zeo:
                return ep

        # 4) 按 LLM 排名顺序对应（fallback）
        rank = candidate.get("rank", 0)
        if 1 <= rank <= len(evidence_packages):
            return evidence_packages[rank - 1]

        return None

    @staticmethod
    def _score_evidence(evidence_type: str) -> int:
        mapping = {
            "direct": 5,
            "near_direct": 4,
            "indirect": 3,
            "similar_guest": 3,
            "single_side": 2,
            "weak": 1,
        }
        return mapping.get(evidence_type, 1)

    @staticmethod
    def _score_diffusion(records: list[dict[str, Any]],
                         guest_a: str = "", guest_b: str = "") -> int:
        """根据两种目标客体分子在该分子筛中的扩散系数差距来打分。

        按相同实验条件（方法 + 温度）分组，每组内分别计算两种客体的平均 D，
        再算比值。最终取各组比值的中位数——避免不同方法/温度下的 D 绝对值
        跨数量级差异（如 MD 模拟 vs 膜渗透实验）污染比较结果。
        """
        ga = guest_a.strip().lower()
        gb = guest_b.strip().lower()

        # 按 (method_norm, temperature_norm) 分组
        from collections import defaultdict
        groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: {ga: [], gb: []}
        )

        for record in records:
            guest = str(record.get("guest_norm") or "").strip().lower()
            value = record.get("diffusion_value_num")
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            if guest not in (ga, gb):
                continue
            method = str(record.get("method_norm") or "").strip()
            temp = str(record.get("temperature_norm") or "").strip()
            groups[(method, temp)][guest].append(float(value))

        # 每组至少两种客体都有数据，用中位数（抗离群值）
        ratios = []
        for guest_vals in groups.values():
            if len(guest_vals[ga]) < 1 or len(guest_vals[gb]) < 1:
                continue
            import statistics
            med_a = statistics.median(guest_vals[ga])
            med_b = statistics.median(guest_vals[gb])
            if med_a > 0 and med_b > 0:
                ratios.append(max(med_a, med_b) / min(med_a, med_b))

        if not ratios:
            return 0

        # 取各组比值的中位数（再次抗离群值）
        ratio = statistics.median(ratios)

        if ratio > 50:
            return 5
        if ratio > 10:
            return 4
        if ratio > 3:
            return 3
        if ratio > 1.3:
            return 2
        return 0

    @staticmethod
    def _score_modification(candidate: dict[str, Any], package: Any) -> int:
        modification = str(candidate.get("modification", "")).strip()
        if not modification:
            return 0
        if package and package.evidence_type in {"direct", "near_direct"}:
            return 2
        if package and package.evidence_type == "similar_guest":
            return 1
        return 1

    @staticmethod
    def _score_mechanism(mechanisms: list[str], reasoning: str) -> int:
        if mechanisms and len(mechanisms) >= 2 and len(reasoning) > 120:
            return 3
        if mechanisms and reasoning:
            return 2
        if mechanisms or reasoning:
            return 1
        return 0


# ============================================================
# 报告写入器
# ============================================================


class ReportWriter:
    def write(
        self,
        guest_a: str,
        guest_b: str,
        result: dict[str, Any],
        score_table: list[dict[str, Any]],
        output_path: Path | None = None,
    ) -> Path:
        output = output_path or REPORT_FILE
        output.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Zeolite Separation Recommendation Report",
            "",
            f"- Guest A: `{guest_a}`",
            f"- Guest B: `{guest_b}`",
            f"- Overall warning: {result.get('overall_warning', 'None')}",
            "",
            "## Ranked Candidates",
            "",
        ]

        for candidate in result.get("ranked_candidates", []):
            lines.extend(
                [
                    f"### Rank {candidate['rank']}: {candidate['zeolite']} ({candidate.get('modification') or 'unmodified'})",
                    f"- Preferred guest: {candidate['preferred_guest']}",
                    f"- Evidence type: {candidate['evidence_type']}",
                    f"- Confidence: {candidate['confidence']}",
                    f"- Mechanisms: {', '.join(candidate.get('separation_mechanism', [])) or 'N/A'}",
                    f"- Database evidence summary: {candidate.get('database_evidence_summary', '')}",
                    f"- Reasoning: {candidate.get('reasoning', '')}",
                    "",
                ]
            )

        if score_table:
            lines.extend(["## Rule-based Scores", ""])
            for item in score_table:
                lines.append(
                    f"- {item['zeolite']} ({item.get('modification') or 'unmodified'}): total={item['total_score']}, "
                    f"evidence={item['evidence_score']}, diffusion={item['diffusion_score']}, "
                    f"modification={item['modification_score']}, mechanism={item['mechanism_score']}"
                )
        else:
            lines.extend(["## Rule-based Scores", "", "- No scores available."])

        if result.get("not_recommended"):
            lines.extend(["", "## Not Recommended", ""])
            for item in result["not_recommended"]:
                lines.append(f"- {item.get('zeolite', 'unknown')}: {item.get('reason', '')}")

        output.write_text("\n".join(lines), encoding="utf-8")
        return output


# ============================================================
# Fallback（LLM 不可用时的推理逻辑）
# ============================================================


def summarize_package(package: EvidencePackage) -> str:
    records = package.records
    guest_counts: dict[str, int] = {}
    diffusion_values: list[float] = []
    methods: set[str] = set()

    for record in records:
        guest = str(record.get("guest_molecule") or record.get("guest_norm") or "").strip()
        if guest:
            guest_counts[guest] = guest_counts.get(guest, 0) + 1
        value = record.get("diffusion_value_num")
        if isinstance(value, (int, float)) and value > 0:
            diffusion_values.append(float(value))
        method = str(record.get("experimental_method") or "").strip()
        if method:
            methods.add(method)

    summary_parts = [f"{len(records)} retrieved record(s)"]
    if guest_counts:
        guest_part = ", ".join(f"{name}: {count}" for name, count in sorted(guest_counts.items()))
        summary_parts.append(f"guest coverage [{guest_part}]")
    if diffusion_values:
        summary_parts.append(
            f"diffusion range {min(diffusion_values):.3e} to {max(diffusion_values):.3e}"
        )
    if methods:
        summary_parts.append(f"methods: {', '.join(sorted(methods)[:3])}")
    return "; ".join(summary_parts)


def infer_mechanisms(package: EvidencePackage) -> list[str]:
    mechanisms: list[str] = []
    records = package.records
    modifications = {str(record.get("modified_ion") or "").strip() for record in records if record.get("modified_ion")}
    guest_names = {str(record.get("guest_molecule") or "").lower() for record in records}

    diffusion_values = [record.get("diffusion_value_num") for record in records if isinstance(record.get("diffusion_value_num"), (int, float))]
    positive_values = [float(value) for value in diffusion_values if value and value > 0]
    if len(positive_values) >= 2 and max(positive_values) / min(positive_values) >= 3:
        mechanisms.append("diffusion selectivity")
    if modifications:
        mechanisms.append("cation or modification effect")
    if any(name in guest_names for name in {"water", "methanol", "ethanol", "co2"}):
        mechanisms.append("adsorption selectivity")
    if any(name in guest_names for name in {"n-butane", "isobutane", "propane", "propylene"}):
        mechanisms.append("size or shape selectivity")
    return mechanisms or ["mechanistic inference required"]


def build_fallback_reasoning(package: EvidencePackage, guest_a: str, guest_b: str) -> str:
    base = (
        f"Fallback mode found {package.evidence_type} evidence for {guest_a} and {guest_b} "
        f"under zeolite key {package.zeolite_key}."
    )
    if package.evidence_type == "direct":
        return base + " Both guests appear in the same zeolite entry group, so this candidate is more defensible."
    if package.evidence_type == "near_direct":
        return base + " Both guests appear under the same zeolite, but conditions vary across records."
    if package.evidence_type == "similar_guest":
        return base + " Recommendation depends on similar-guest analogy rather than full direct evidence."
    if package.evidence_type == "single_side":
        return base + " Only one target guest has direct records, so the recommendation is weaker."
    return base + " Evidence is limited and should be treated cautiously."


def build_fallback_result(guest_a: str, guest_b: str, evidence_packages: list[EvidencePackage]) -> dict[str, Any]:
    result = {**FALLBACK_RESULT_TEMPLATE}
    result["input_guests"] = {"guest_A": guest_a, "guest_B": guest_b}
    if not evidence_packages:
        result["overall_warning"] = (
            "No defensible database evidence was found for this guest pair. "
            "Fallback mode cannot provide a strong recommendation."
        )
        return result

    for index, package in enumerate(evidence_packages, start=1):
        first_record = package.records[0] if package.records else {}
        modification = first_record.get("modified_ion")
        if not isinstance(modification, str):
            modification = ""

        matched_text = ", ".join(package.matched_guests) or "undetermined"
        summary = summarize_package(package)
        result["ranked_candidates"].append(
            {
                "rank": index,
                "zeolite": first_record.get("zeolite_name") or package.zeolite_key.split("__")[0],
                "modification": modification,
                "preferred_guest": matched_text,
                "evidence_type": package.evidence_type,
                "separation_mechanism": infer_mechanisms(package),
                "database_evidence_summary": summary,
                "reasoning": build_fallback_reasoning(package, guest_a, guest_b),
                "confidence": EVIDENCE_CONFIDENCE_MAP.get(package.evidence_type, "low"),
            }
        )
    return result


# ============================================================
# 主流程
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot zeolite separation recommender")
    parser.add_argument("--guest_a", help="First guest molecule")
    parser.add_argument("--guest_b", help="Second guest molecule")
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT, help="Number of candidates to keep")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "prediction_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label added to console output for batch runs",
    )
    return parser.parse_args()


def prompt_for_missing_args(args: argparse.Namespace) -> tuple[str, str]:
    guest_a = args.guest_a or input("请输入第一个客体分子: ").strip()
    guest_b = args.guest_b or input("请输入第二个客体分子: ").strip()
    if not guest_a or not guest_b:
        raise ValueError("Both guest molecules are required.")
    return guest_a, guest_b


def run() -> Path:
    args = parse_args()
    guest_a, guest_b = prompt_for_missing_args(args)

    repository = load_data()
    llm_client = LLMClient()
    enricher = MetadataEnricher(llm_client)
    enricher.ensure_metadata(repository)

    retriever = EvidenceRetriever(repository, enricher, llm_client)
    evidence_packages = retriever.retrieve(guest_a, guest_b, args.top_k)

    guest_a_metadata = enricher.get_molecule_metadata(guest_a)
    guest_b_metadata = enricher.get_molecule_metadata(guest_b)
    zeolite_metadata_map = {
        package.zeolite_key: enricher.get_zeolite_metadata(package.zeolite_key, repository)
        for package in evidence_packages
    }

    if llm_client.enabled and evidence_packages:
        try:
            prompt_builder = PromptBuilder()
            separation_prompt = prompt_builder.build_separation_prompt(
                guest_a,
                guest_b,
                guest_a_metadata,
                guest_b_metadata,
                evidence_packages,
                zeolite_metadata_map,
            )
            raw_result = llm_client.generate_json(SYSTEM_ROLE_PROMPT, separation_prompt)
            parsed_result = ResultParser().parse(raw_result)
        except Exception as exc:
            parsed_result = build_fallback_result(guest_a, guest_b, evidence_packages)
            parsed_result["overall_warning"] = (
                "LLM reasoning failed and fallback mode was used. "
                f"Failure detail: {exc}"
            )
    else:
        parsed_result = build_fallback_result(guest_a, guest_b, evidence_packages)

    score_table = Scorer().score(parsed_result, evidence_packages, guest_a, guest_b)
    if score_table:
        score_lookup = {item["zeolite_key"]: item["total_score"] for item in score_table}
        parsed_result["ranked_candidates"].sort(
            key=lambda candidate: score_lookup.get(
                (
                    f"{str(candidate.get('zeolite', '')).strip().lower()}__{str(candidate.get('modification', '')).strip().lower()}"
                    if candidate.get("modification")
                    else str(candidate.get("zeolite", "")).strip().lower()
                ),
                0,
            ),
            reverse=True,
        )
        for index, candidate in enumerate(parsed_result["ranked_candidates"], start=1):
            candidate["rank"] = index

    report_path = ReportWriter().write(guest_a, guest_b, parsed_result, score_table, args.output)
    if args.label:
        print(f"=== {args.label} ===")
    try:
        print(json.dumps(parsed_result, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(parsed_result, ensure_ascii=True, indent=2))
    print(f"\nReport written to: {report_path}")
    return report_path


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    run()
