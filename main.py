import argparse
import json
import os
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - fallback when PyYAML missing
    yaml = None

from agents.agent_manager import AgentManager
from aggregation.aggregator import aggregate
from retrieval.bm25_client import BM25Client
from scoring.scorer import score_answers
from utils.logger import TraceLogger
from utils.schemas import FinalOutput, Question


DEFAULT_CONFIG: Dict[str, Any] = {
    "MODELS": {
        "agent_A": "deepseek-ai/DeepSeek-V2.5",
        "agent_B": "Qwen/Qwen2.5-7B-Instruct",
        "agent_C": "THUDM/glm-4-9b-chat",
        "agent_D": "gpt-4o-mini",
    },
    "SEARCH_BACKEND": {"type": "BM25", "index_path": "./data/hotpotqa_corpus.json", "max_docs": 50000},
    "PARALLELISM": {"agent_workers": 4},
    "OUTPUT": {
        "save_traces": True,
        "save_agent_reasoning": True,
        "save_evidence": True,
    },
    "API": {"base_url": "https://api.siliconflow.cn/v1/chat/completions", "api_key_env": "DEEPSEEK_API_KEY", "timeout": 60},
    "RETRIEVAL": {"k": 8},
    "PARTIAL_DECENTRALIZATION": {"enabled": True, "cluster_k": 3},
}


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if yaml is None or not os.path.exists(path):
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except Exception:
            return DEFAULT_CONFIG
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg)
    return merged


def run_pipeline(question: str, config_path: str = "config.yaml", mode: Optional[str] = None) -> FinalOutput:
    config = load_config(config_path)
    q = Question(id=str(uuid.uuid4()), text=question)
    search_cfg = config.get("SEARCH_BACKEND", {})
    index_path = search_cfg.get("index_path") if isinstance(search_cfg, dict) else None
    max_docs = search_cfg.get("max_docs", 50000) if isinstance(search_cfg, dict) else 50000
    retrieval_client = BM25Client(index_path=index_path, max_docs=max_docs)
    models = config.get("MODELS", DEFAULT_CONFIG["MODELS"])
    parallelism = config.get("PARALLELISM", {}).get("agent_workers", 4)
    api_config = config.get("API", DEFAULT_CONFIG.get("API", {}))
    retrieval_k = config.get("RETRIEVAL", {}).get("k", 8)
    cluster_k = config.get("PARTIAL_DECENTRALIZATION", {}).get("cluster_k", 3)
    partial_mode = config.get("PARTIAL_DECENTRALIZATION", {}).get("enabled", True)

    manager = AgentManager(
        models=models,
        retrieval_client=retrieval_client,
        max_workers=parallelism,
        api_config=api_config,
        retrieval_k=retrieval_k,
        cluster_k=cluster_k,
    )
    # Determine mode: if explicit mode provided, honor it; otherwise use config partial flag
    if mode is None:
        mode_to_run = 'partial' if partial_mode else 'agents'
    else:
        mode_to_run = mode

    if mode_to_run == 'debate':
        agent_answers, agent_queries, shared_evidence, meta = manager.run_multi_round_debate(q)
    elif mode_to_run == 'partial':
        agent_answers, agent_queries, shared_evidence, meta = manager.run_partial_decentralized(q)
    else:
        # fallback: run simple agents (no shared retrieval)
        agent_answers = manager.run_agents(q)
        shared_evidence = None
        agent_queries = None
        meta = None
    scored = score_answers(agent_answers)
    answers_by_id = {ans.agent_id: ans for ans in agent_answers}
    final = aggregate(scored, answers_by_id)

    logger = TraceLogger(config.get("OUTPUT", {}))
    logger.persist(q, agent_answers, scored, final, metadata=meta)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MVP agent pipeline interactively or on HotpotQA datasets")
    parser.add_argument("--input-file", "-i", default="./data/hotpot_train_v1.1.json", help="Path to HotpotQA json file")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Randomly sample N questions from dataset")
    parser.add_argument("--batch", action="store_true", help="Run pipeline in batch over dataset (or sampled subset)")
    parser.add_argument("--out", "-o", default=None, help="Output file for batch results. Default: logs/<timestamp>_results.(json|jsonl)")
    parser.add_argument("--format", "-f", choices=["json", "jsonl"], default="json", help="Output format for batch results")
    parser.add_argument("--rate", "-r", type=float, default=0.0, help="Seconds to wait between requests (rate limit)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["partial", "debate", "agents"], default=None, help="Which pipeline mode to run (overrides config)")
    args = parser.parse_args()

    def load_hotpot(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # common HotpotQA formats: list of dicts, or dict with 'data' key
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                entries = data["data"]
            elif "questions" in data and isinstance(data["questions"], list):
                entries = data["questions"]
            else:
                # maybe a mapping of id->entry
                entries = list(data.values())
        elif isinstance(data, list):
            entries = data
        else:
            entries = []
        return entries

    def extract_question(entry: Dict[str, Any]) -> Optional[str]:
        for k in ("question", "query", "question_text", "text"):
            if isinstance(entry, dict) and k in entry and isinstance(entry[k], str):
                return entry[k]
        # sometimes 'question' nested under other keys
        if isinstance(entry, dict) and "meta" in entry and isinstance(entry["meta"], dict):
            return entry["meta"].get("question")
        return None

    def sample_entries(entries: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if k <= 0 or k >= len(entries):
            return entries
        return random.sample(entries, k)

    def run_batch(entries: List[Dict[str, Any]], out_path: str, config_path: str, rate: float = 0.0, fmt: str = "json"):
        results: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries, start=1):
            qtext = extract_question(entry)
            if not qtext:
                continue
            try:
                final = run_pipeline(qtext, config_path)
            except Exception as e:
                result_obj = {"id": entry.get("id"), "question": qtext, "error": str(e)}
            else:
                result_obj = {
                    "id": entry.get("id"),
                    "question": qtext,
                    "answer": getattr(final, "answer", str(final)),
                    "all_agent_scores": getattr(final, "all_agent_scores", None),
                }
            results.append(result_obj)
            if rate and idx < len(entries):
                time.sleep(rate)
        # ensure output directory exists
        out_dir = os.path.dirname(out_path) or "."
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        if fmt == "jsonl":
            with open(out_path, "w", encoding="utf-8") as out_f:
                for r in results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
        return len(results)

    if args.batch:
        entries = load_hotpot(args.input_file)
        if not entries:
            print(f"No entries found in {args.input_file}")
            raise SystemExit(1)
        if args.sample and args.sample > 0:
            entries = sample_entries(entries, args.sample)
        # default output to logs/<timestamp>_results.(json|jsonl)
        out_path = args.out
        if out_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "jsonl" if args.format == "jsonl" else "json"
            out_path = os.path.join("logs", f"{ts}_results.{ext}")
        print(f"Processing {len(entries)} questions; writing results to {out_path}")
        count = run_batch(entries, out_path, args.config, rate=args.rate, fmt=args.format)
        print(f"Finished. Wrote {count} results to {out_path}")
    elif args.sample and args.sample > 0:
        entries = load_hotpot(args.input_file)
        if not entries:
            print(f"No entries found in {args.input_file}")
            raise SystemExit(1)
        entries = sample_entries(entries, args.sample)
        for entry in entries:
            qtext = extract_question(entry)
            if not qtext:
                continue
            print("Question:", qtext)
            res = run_pipeline(qtext, args.config, mode=args.mode)
            print(getattr(res, "answer", res))
            print(getattr(res, "all_agent_scores", None))
            if args.rate and args.rate > 0:
                time.sleep(args.rate)
            print("---")
    else:
        user_question = input("Enter HotpotQA question: ")
        result = run_pipeline(user_question, args.config, mode=args.mode)
        print(result.answer)
        print(result.all_agent_scores)
