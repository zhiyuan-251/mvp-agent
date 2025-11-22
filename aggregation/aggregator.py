from typing import Dict, List

from utils.schemas import AgentAnswer, FinalOutput, ScoredAnswer


def aggregate(scored: List[ScoredAnswer], agent_answers: Dict[str, AgentAnswer]) -> FinalOutput:
    if not scored:
        return FinalOutput(
            answer="No answers produced.",
            winner_agent="",
            all_agent_scores={},
            debug_traces={},
        )
    best = max(scored, key=lambda s: s.score)
    scores = {s.agent_id: s.score for s in scored}
    winner_answer = agent_answers.get(best.agent_id)
    final_answer = winner_answer.answer if winner_answer else ""
    debug = {
        "agent_reasoning": {aid: ans.reasoning_chain for aid, ans in agent_answers.items()},
        "agent_answers": {aid: ans.answer for aid, ans in agent_answers.items()},
    }
    return FinalOutput(
        answer=final_answer,
        winner_agent=best.agent_id,
        all_agent_scores=scores,
        debug_traces=debug,
    )
