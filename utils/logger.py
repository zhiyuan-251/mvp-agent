import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from utils.schemas import AgentAnswer, FinalOutput, Question, ScoredAnswer


class TraceLogger:
    def __init__(self, output_config: Optional[Dict] = None, base_dir: str = "logs"):
        cfg = output_config or {}
        self.save_traces = cfg.get("save_traces", True)
        self.save_agent_reasoning = cfg.get("save_agent_reasoning", True)
        self.save_evidence = cfg.get("save_evidence", True)
        self.base_dir = base_dir

    def persist(
        self,
        question: Question,
        agent_answers: List[AgentAnswer],
        scored_answers: List[ScoredAnswer],
        final_output: FinalOutput,
        metadata: Optional[Dict] = None,
    ) -> None:
        if not self.save_traces:
            return
        os.makedirs(self.base_dir, exist_ok=True)
        # 使用时间戳命名文件，格式：YYYYMMDD_HHMMSS.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.base_dir, f"{timestamp}.json")
        payload = {
            "question": question.text,
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "scores": {s.agent_id: s.score for s in scored_answers},
            "winner": final_output.winner_agent,
        }
        if metadata:
            payload["meta"] = metadata
        for ans in agent_answers:
            entry = {"answer": ans.answer}
            if self.save_agent_reasoning:
                entry["reasoning_chain"] = ans.reasoning_chain
            if self.save_evidence:
                entry["evidence"] = [
                    {"doc_id": d.doc_id, "score": d.score, "text": d.text}
                    for d in (ans.evidence.top_docs if ans.evidence else [])
                ]
            payload["agents"][ans.agent_id] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
