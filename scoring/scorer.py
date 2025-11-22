from typing import Dict, List

from utils.schemas import AgentAnswer, ScoredAnswer


def _tokenize(text: str) -> set:
    return {t for t in text.lower().split() if t}


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    union = len(ta | tb)
    return overlap / union if union else 0.0


def _inter_agent_agreement(target: AgentAnswer, all_answers: List[AgentAnswer]) -> float:
    if len(all_answers) <= 1:
        return 0.0
    sims = []
    for other in all_answers:
        if other.agent_id == target.agent_id:
            continue
        sims.append(_similarity(target.answer, other.answer))
    return sum(sims) / len(sims) if sims else 0.0


def score_answers(agent_answers: List[AgentAnswer]) -> List[ScoredAnswer]:
    scored: List[ScoredAnswer] = []
    evidence_text: Dict[str, str] = {
        ans.agent_id: " ".join(doc.text for doc in (ans.evidence.top_docs if ans.evidence else []))
        for ans in agent_answers
    }

    for ans in agent_answers:
        ev_text = evidence_text.get(ans.agent_id, "")
        evidence_consistency = _similarity(ans.answer, ev_text)
        chain_coherence = min(len(ans.reasoning_chain) / 500, 1.0)
        inter_agent_agreement = _inter_agent_agreement(ans, agent_answers)
        hallucination_penalty = 0.3 if not ans.references else max(0.0, 0.3 - evidence_consistency)
        score = (
            0.40 * evidence_consistency
            + 0.25 * chain_coherence
            + 0.20 * inter_agent_agreement
            - 0.15 * hallucination_penalty
        )
        scored.append(
            ScoredAnswer(
                agent_id=ans.agent_id,
                score=score,
                evidence_consistency=evidence_consistency,
                chain_coherence=chain_coherence,
                inter_agent_agreement=inter_agent_agreement,
                hallucination_penalty=hallucination_penalty,
            )
        )
    return scored
