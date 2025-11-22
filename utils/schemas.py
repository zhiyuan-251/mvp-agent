from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Question:
    id: str
    text: str


@dataclass
class AgentQuery:
    agent_id: str
    question: str
    sub_questions: List[str] = field(default_factory=list)
    retrieval_queries: List[str] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    text: str
    score: float


@dataclass
class Evidence:
    agent_id: str
    top_docs: List[Document] = field(default_factory=list)


@dataclass
class AgentAnswer:
    agent_id: str
    reasoning_chain: str
    answer: str
    uncertainty: float
    references: List[str] = field(default_factory=list)
    evidence: Optional[Evidence] = None


@dataclass
class ScoredAnswer:
    agent_id: str
    score: float
    evidence_consistency: float
    chain_coherence: float
    inter_agent_agreement: float
    hallucination_penalty: float


@dataclass
class FinalOutput:
    answer: str
    winner_agent: str
    all_agent_scores: Dict[str, float]
    debug_traces: Optional[Dict[str, Any]] = None
