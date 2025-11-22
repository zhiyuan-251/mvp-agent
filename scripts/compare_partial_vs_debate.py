"""Small-scale evaluator to compare Partial Decentralization vs Multi-Round Debate.

Usage:
    python scripts/compare_partial_vs_debate.py --input data/hotpot_test_fullwiki_v1.json --sample 20 --out logs/eval_compare.jsonl

What it measures (per question):
- question id/text
- partial: consensus queries, #shared_docs, top agent scores, winner, answer
- debate: consensus queries, #shared_docs, top agent scores, winner, answer
- simple reference match (if dataset contains an 'answer' or 'answer_text' field)

This script uses the existing AgentManager and pipeline functions.
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

# ensure repo root is on sys.path so we can import main and package modules
import sys
sys.path.insert(0, os.getcwd())

from main import load_config
from retrieval.bm25_client import BM25Client
from agents.agent_manager import AgentManager
from scoring.scorer import score_answers
from utils.schemas import Question


def load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # HotpotQA格式：直接是包含问题字典的列表
    if isinstance(data, list):
        print(f"Loaded {len(data)} entries from {path}")
        return data
    
    # 其他可能的格式
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            print(f"Loaded {len(data['data'])} entries from {path} (nested in 'data')")
            return data['data']
        if 'questions' in data and isinstance(data['questions'], list):
            print(f"Loaded {len(data['questions'])} entries from {path} (nested in 'questions')")
            return data['questions']
        print(f"Converting dict to list: {len(data)} entries")
        return list(data.values())
    
    print(f"Loaded data of type {type(data)}")
    return data


def extract_question_text(entry: Dict[str, Any]) -> Optional[str]:
    for k in ('question', 'query', 'question_text', 'text'):
        if isinstance(entry, dict) and k in entry and isinstance(entry[k], str):
            return entry[k]
    if isinstance(entry, dict) and 'meta' in entry and isinstance(entry['meta'], dict):
        return entry['meta'].get('question')
    return None


def simple_answer_match(pred: str, golds: List[str]) -> bool:
    if not pred or not golds:
        return False
    pred_norm = pred.strip().lower()
    for g in golds:
        if not g:
            continue
        if g.strip().lower() in pred_norm or pred_norm in g.strip().lower():
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/hotpot_test_fullwiki_v1.json')
    parser.add_argument('--sample', '-s', type=int, default=100, help='Number of questions to sample')
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['both', 'partial', 'debate'], default='both', help='Which mode(s) to run')
    args = parser.parse_args()

    cfg = load_config()
    search_cfg = cfg.get('SEARCH_BACKEND', {})
    index_path = search_cfg.get('index_path') if isinstance(search_cfg, dict) else None
    max_docs = search_cfg.get('max_docs', 50000) if isinstance(search_cfg, dict) else 50000

    client = BM25Client(index_path=index_path, max_docs=max_docs)
    models = cfg.get('MODELS')
    manager = AgentManager(models=models, retrieval_client=client, max_workers=cfg.get('PARALLELISM', {}).get('agent_workers', 4), api_config=cfg.get('API', {}), retrieval_k=cfg.get('RETRIEVAL', {}).get('k', 8), cluster_k=cfg.get('PARTIAL_DECENTRALIZATION', {}).get('cluster_k', 3))

    data = load_dataset(args.input)
    if args.sample > 0 and args.sample < len(data):
        random.seed(args.seed)
        entries = random.sample(data, args.sample)
    else:
        entries = data

    out_path = args.out or os.path.join('logs', f'eval_compare_{int(time.time())}.jsonl')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    stats = {
        'n': 0,
        'partial_winner_counts': {},
        'debate_winner_counts': {},
        'partial_shared_docs_avg': 0.0,
        'debate_shared_docs_avg': 0.0,
        'partial_match_count': 0,
        'debate_match_count': 0,
        'partial_avg_f1': 0.0,
        'debate_avg_f1': 0.0,
        'partial_avg_time_s': 0.0,
        'debate_avg_time_s': 0.0,
        'partial_avg_llm_calls': 0.0,
        'debate_avg_llm_calls': 0.0,
        'partial_avg_dominance': 0.0,
        'debate_avg_dominance': 0.0,
    }

    with open(out_path, 'w', encoding='utf-8') as outf:
        for entry in entries:
            qtext = extract_question_text(entry)
            if not qtext:
                continue
            qid = entry.get('id') or entry.get('question_id') or entry.get('qid') or f'idx_{stats["n"]}'

            # Partial (if requested)
            q = Question(id=str(qid), text=qtext)
            partial_ans = partial_queries = partial_ev = partial_meta = None
            partial_scores = []
            partial_winner = None
            partial_f1 = None
            partial_time = None
            partial_llm_calls = None
            if args.mode in ('both', 'partial'):
                try:
                    from utils.llm_client import get_and_reset_call_count
                except Exception:
                    def get_and_reset_call_count():
                        return 0
                # reset counter
                get_and_reset_call_count()
                t0 = time.time()
                try:
                    partial_ans, partial_queries, partial_ev, partial_meta = manager.run_partial_decentralized(q)
                except Exception as e:
                    print('Partial run error for', qid, e)
                    continue
                partial_time = time.time() - t0
                partial_llm_calls = get_and_reset_call_count()
                partial_scores = score_answers(partial_ans)
                partial_winner = partial_scores[0].agent_id if partial_scores else None

            # Debate (if requested)
            debate_ans = debate_queries = debate_ev = debate_meta = None
            debate_scores = []
            debate_winner = None
            debate_f1 = None
            debate_time = None
            debate_llm_calls = None
            if args.mode in ('both', 'debate'):
                try:
                    from utils.llm_client import get_and_reset_call_count
                except Exception:
                    def get_and_reset_call_count():
                        return 0
                get_and_reset_call_count()
                t0 = time.time()
                try:
                    debate_ans, debate_queries, debate_ev, debate_meta = manager.run_multi_round_debate(q)
                except Exception as e:
                    print('Debate run error for', qid, e)
                    continue
                debate_time = time.time() - t0
                debate_llm_calls = get_and_reset_call_count()
                debate_scores = score_answers(debate_ans)
                debate_winner = debate_scores[0].agent_id if debate_scores else None

            # Enhanced gold answer extraction for HotpotQA
            def extract_gold_answers(entry: Dict[str, Any]) -> List[str]:
                golds = []
                
                # 1. 直接检查常见答案字段
                for k in ('answer', 'answer_text', 'final_answer', 'gold_answer'):
                    v = entry.get(k)
                    if isinstance(v, str) and v.strip():
                        golds.append(v.strip())
                
                # 2. 检查嵌套答案（某些格式可能将答案放在子字典中）
                if not golds and isinstance(entry, dict):
                    for key, val in entry.items():
                        if isinstance(val, dict) and 'answer' in val:
                            a = val.get('answer')
                            if isinstance(a, str) and a.strip():
                                golds.append(a.strip())
                
                # 3. 检查是否有答案列表（多个可能的正确答案）
                for k in ('answers', 'gold_answers', 'possible_answers'):
                    v = entry.get(k)
                    if isinstance(v, list):
                        for ans in v:
                            if isinstance(ans, str) and ans.strip():
                                golds.append(ans.strip())
                            elif isinstance(ans, dict) and 'text' in ans:
                                text = ans.get('text')
                                if isinstance(text, str) and text.strip():
                                    golds.append(text.strip())
                
                return list(set(golds))  # 去重
            
            golds = extract_gold_answers(entry)
            
            # 调试信息：如果没有找到答案，打印条目结构
            if not golds:
                print(f"Warning: No gold answer found for question {qid}")
                print(f"Entry keys: {list(entry.keys()) if isinstance(entry, dict) else 'Not a dict'}")
                if isinstance(entry, dict):
                    for k in ['answer', '_id', 'question']:
                        if k in entry:
                            print(f"  {k}: {entry[k]}")

            partial_pred = partial_ans[0].answer if partial_ans else ''
            debate_pred = debate_ans[0].answer if debate_ans else ''

            # compute F1 (token-level) for predictions against golds
            def token_f1(a: str, b: str) -> float:
                import re
                def normalize(s: str):
                    s = s or ''
                    s = s.lower()
                    s = re.sub(r"[^a-z0-9]+", ' ', s)
                    toks = [t for t in s.split() if t]
                    return toks
                ta = normalize(a)
                tb = normalize(b)
                if not ta or not tb:
                    return 0.0
                common = set(ta) & set(tb)
                if not common:
                    return 0.0
                prec = len(common) / len(ta)
                rec = len(common) / len(tb)
                if prec + rec == 0:
                    return 0.0
                return 2 * prec * rec / (prec + rec)

            def best_f1(pred: str, golds_list: List[str]) -> float:
                if not golds_list:
                    return 0.0
                best = 0.0
                for g in golds_list:
                    f = token_f1(pred, g)
                    if f > best:
                        best = f
                return best

            p_match = simple_answer_match(partial_pred, golds)
            d_match = simple_answer_match(debate_pred, golds)
            partial_f1 = best_f1(partial_pred, golds) if partial_pred else 0.0
            debate_f1 = best_f1(debate_pred, golds) if debate_pred else 0.0

            # collect per-entry result
            # compute dominance metric: top - second
            def dominance(scores_list):
                if not scores_list:
                    return 0.0
                vals = sorted([s.score for s in scores_list], reverse=True)
                if len(vals) == 1:
                    return vals[0]
                return vals[0] - vals[1]

            rec = {
                'id': qid,
                'question': qtext,
                'partial': {
                    'consensus_queries': partial_meta.get('consensus_queries') if partial_meta else None,
                    'shared_docs': len(partial_ev.top_docs) if partial_ev else 0,
                    'top_scores': {s.agent_id: s.score for s in partial_scores},
                    'winner': partial_winner,
                    'answer': partial_pred,
                    'match': p_match,
                    'f1': partial_f1,
                    'time_s': partial_time,
                    'llm_calls': partial_llm_calls,
                    'dominance': dominance(partial_scores),
                },
                'debate': {
                    'consensus_queries': debate_meta.get('consensus_queries') if debate_meta else None,
                    'shared_docs': len(debate_ev.top_docs) if debate_ev else 0,
                    'top_scores': {s.agent_id: s.score for s in debate_scores},
                    'winner': debate_winner,
                    'answer': debate_pred,
                    'match': d_match,
                    'f1': debate_f1,
                    'time_s': debate_time,
                    'llm_calls': debate_llm_calls,
                    'dominance': dominance(debate_scores),
                },
                'gold_answers': golds,
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')

            # streaming stats
            stats['n'] += 1
            stats['partial_winner_counts'][partial_winner] = stats['partial_winner_counts'].get(partial_winner, 0) + 1
            stats['debate_winner_counts'][debate_winner] = stats['debate_winner_counts'].get(debate_winner, 0) + 1
            stats['partial_shared_docs_avg'] += rec['partial']['shared_docs']
            stats['debate_shared_docs_avg'] += rec['debate']['shared_docs']
            stats['partial_match_count'] += 1 if p_match else 0
            stats['debate_match_count'] += 1 if d_match else 0
            stats['partial_avg_f1'] += rec['partial'].get('f1', 0.0) or 0.0
            stats['debate_avg_f1'] += rec['debate'].get('f1', 0.0) or 0.0
            stats['partial_avg_time_s'] += rec['partial'].get('time_s') or 0.0
            stats['debate_avg_time_s'] += rec['debate'].get('time_s') or 0.0
            stats['partial_avg_llm_calls'] += rec['partial'].get('llm_calls') or 0.0
            stats['debate_avg_llm_calls'] += rec['debate'].get('llm_calls') or 0.0
            stats['partial_avg_dominance'] += rec['partial'].get('dominance') or 0.0
            stats['debate_avg_dominance'] += rec['debate'].get('dominance') or 0.0

    # finalize averages
    if stats['n'] > 0:
        stats['partial_shared_docs_avg'] /= stats['n']
        stats['debate_shared_docs_avg'] /= stats['n']
        stats['partial_avg_f1'] /= stats['n']
        stats['debate_avg_f1'] /= stats['n']
        stats['partial_avg_time_s'] /= stats['n']
        stats['debate_avg_time_s'] /= stats['n']
        stats['partial_avg_llm_calls'] /= stats['n']
        stats['debate_avg_llm_calls'] /= stats['n']
        stats['partial_avg_dominance'] /= stats['n']
        stats['debate_avg_dominance'] /= stats['n']

    summary_path = out_path + '.summary.json'
    with open(summary_path, 'w', encoding='utf-8') as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print('Wrote detailed results to', out_path)
    print('Summary written to', summary_path)


if __name__ == '__main__':
    main()
