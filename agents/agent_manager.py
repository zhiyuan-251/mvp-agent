from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from agents.agent_worker import SingleAgentWorker
from aggregation.subquery_cluster import cluster_subqueries
from retrieval.bm25_client import BM25Client
from utils.schemas import AgentAnswer, AgentQuery, Evidence, Question


class AgentManager:
    def __init__(
        self,
        models: Dict[str, str],
        retrieval_client: BM25Client,
        max_workers: int = 4,
        api_config: Dict = None,
        retrieval_k: int = 8,
        cluster_k: int = 3,
    ):
        self.models = models
        self.retrieval_client = retrieval_client
        self.max_workers = max_workers
        self.api_config = api_config or {}
        self.retrieval_k = retrieval_k
        self.cluster_k = cluster_k

    def run_agents(self, question: Question) -> List[AgentAnswer]:
        answers: List[AgentAnswer] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    SingleAgentWorker(
                        agent_id,
                        model_name,
                        self.retrieval_client,
                        self.api_config,
                        self.retrieval_k,
                    ),
                    question,
                ): agent_id
                for agent_id, model_name in self.models.items()
            }
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    answers.append(future.result())
                except Exception as exc:  # pragma: no cover - guardrail
                    answers.append(
                        AgentAnswer(
                            agent_id=agent_id,
                            reasoning_chain=f"Agent {agent_id} failed: {exc}",
                            answer="Error during generation",
                            uncertainty=1.0,
                            references=[],
                            evidence=None,
                        )
                    )
        return answers

    def run_partial_decentralized(self, question: Question):
        print("\n🔥 Step 1：每个 solver 自己拆分（local decomposition）\nCollecting decompositions from agents...")
        agent_queries = self._collect_decompositions(question)
        print(f"Collected decompositions from {len(agent_queries)} agents.")

        print("\n🔥 Step 2：聚合所有 sub-queries 并进行 voting / clustering\nClustering sub-queries to form consensus and edge cases...")
        consensus_queries, clusters = self._consensus_subqueries(agent_queries)
        print(f"Consensus sub-queries: {consensus_queries}")

        print("\n🔥 Step 3：统一检索（使用 consensus sub-queries + 各 agent retrieval_queries）\nRunning shared retrieval for consensus sub-queries and agents' retrieval queries (including edge-cluster special retrieval)...")
        shared_evidence = self._shared_retrieval(consensus_queries, question.text, agent_queries, clusters)
        print(f"Retrieved {len(shared_evidence.top_docs)} shared evidence docs.")

        print("\n🔥 Step 4：让所有 agent 基于统一 evidence 回答（并保留 scoring）\nCollecting agent answers using shared evidence...")
        answers = self._collect_answers(question, agent_queries, shared_evidence)
        meta = {
            "subqueries_raw": {aq.agent_id: aq.sub_questions for aq in agent_queries},
            "consensus_queries": consensus_queries,
            "clusters": {str(k): v for k, v in clusters.items()},
        }
        return answers, agent_queries, shared_evidence, meta

    def _collect_decompositions(self, question: Question) -> List[AgentQuery]:
        queries: List[AgentQuery] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    SingleAgentWorker(
                        agent_id,
                        model_name,
                        self.retrieval_client,
                        self.api_config,
                        self.retrieval_k,
                    ).decompose,
                    question,
                ): agent_id
                for agent_id, model_name in self.models.items()
            }
            for future in as_completed(futures):
                try:
                    queries.append(future.result())
                except Exception as exc:  # pragma: no cover
                    queries.append(
                        AgentQuery(
                            agent_id=futures[future],
                            question=question.text,
                            sub_questions=[question.text],
                            retrieval_queries=[question.text],
                        )
                    )
        # Log each agent's decomposition briefly
        for aq in queries:
            print(f"- {aq.agent_id} -> {len(aq.sub_questions)} sub-queries: {aq.sub_questions}")
        return queries

    def _consensus_subqueries(self, agent_queries: List[AgentQuery]):
        subqs: List[str] = []
        for aq in agent_queries:
            subqs.extend(aq.sub_questions)
        reps, clusters = cluster_subqueries(subqs, k=self.cluster_k)
        # Print cluster summaries
        try:
            print("Clusters summary:")
            for cid, items in clusters.items():
                sample = items[:5]
                print(f"- cluster {cid}: {len(items)} members; examples: {sample}")
        except Exception:
            pass
        return reps or subqs, clusters

    def run_multi_round_debate(self, question: Question):
        """Run the multi-round debate baseline.

        Rounds:
          1) Each agent generates sub-queries
          2) Each agent critiques other agents' sub-queries
          3) Each agent refines or defends its sub-queries
          4) Consensus clustering and unified retrieval (with edge handling)
          5) Agents answer using shared evidence
        """
        print("\n[Debate] Round 1: Agents generate initial sub-queries")
        initial_queries = self._collect_decompositions(question)
        print(f"[Debate] Collected {len(initial_queries)} initial decompositions.")

        # Prepare workers
        workers = {}
        for agent_id, model_name in self.models.items():
            workers[agent_id] = SingleAgentWorker(agent_id, model_name, self.retrieval_client, self.api_config, self.retrieval_k)

        print("\n[Debate] Round 2: Agents critique others' sub-queries")
        # critiques_targeted[agent_id] = list of critique strings about that agent's subqs
        critiques_targeted = {aq.agent_id: [] for aq in initial_queries}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for critic in initial_queries:
                critic_worker = workers.get(critic.agent_id)
                for target in initial_queries:
                    if critic.agent_id == target.agent_id:
                        continue
                    futures[executor.submit(critic_worker.critique_subqueries, target.agent_id, target.sub_questions, question)] = (critic.agent_id, target.agent_id)
            for fut in as_completed(futures):
                critic_id, target_id = futures[fut]
                try:
                    text = fut.result()
                except Exception:
                    text = ""
                critiques_targeted[target_id].append(f"From {critic_id}: {text}")

        print("\n[Debate] Round 3: Agents refine or defend their sub-queries")
        refined_queries: List[AgentQuery] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for orig in initial_queries:
                worker = workers.get(orig.agent_id)
                crits = critiques_targeted.get(orig.agent_id, [])
                futures[executor.submit(worker.refine_subqueries, orig, crits, question)] = orig.agent_id
            for fut in as_completed(futures):
                try:
                    refined_queries.append(fut.result())
                except Exception:
                    # fallback to original
                    aid = futures[fut]
                    orig = next((x for x in initial_queries if x.agent_id == aid), None)
                    if orig:
                        refined_queries.append(orig)

        # Round 4: clustering + shared retrieval
        print("\n[Debate] Round 4: Consensus clustering and unified retrieval")
        consensus_queries, clusters = self._consensus_subqueries(refined_queries)
        shared_evidence = self._shared_retrieval(consensus_queries, question.text, refined_queries, clusters)

        # Round 5: answers
        print("\n[Debate] Round 5: Agents answer using shared evidence")
        answers = self._collect_answers(question, refined_queries, shared_evidence)

        meta = {
            "initial_subqueries": {aq.agent_id: aq.sub_questions for aq in initial_queries},
            "critiques": critiques_targeted,
            "refined_subqueries": {rq.agent_id: rq.sub_questions for rq in refined_queries},
            "consensus_queries": consensus_queries,
            "clusters": {str(k): v for k, v in clusters.items()},
        }
        return answers, refined_queries, shared_evidence, meta

    def _shared_retrieval(self, subqueries: List[str], fallback_question: str, agent_queries: List[AgentQuery] = None, clusters: dict = None) -> Evidence:
        collected = []
        # Build a hybrid query set: consensus representatives + top agent retrieval queries + fallback question
        queries = list(dict.fromkeys(subqueries + [fallback_question]))
        # include a small set of per-agent retrieval queries to avoid over-centralizing
        try:
            if agent_queries:
                extra = []
                for aq in agent_queries:
                    if hasattr(aq, 'retrieval_queries') and aq.retrieval_queries:
                        extra.extend(aq.retrieval_queries[:2])
                # cap extras to avoid explosion
                if extra:
                    for e in extra:
                        if e not in queries:
                            queries.append(e)
        except Exception:
            pass

        # Primary retrieval pass: retrieve for the hybrid queries
        for q in queries:
            docs = self.retrieval_client.search(q, k=self.retrieval_k)
            collected.extend(docs)

        # Edge-cluster handling: for clusters with very few members (e.g., 1 or 2),
        # perform an extra focused retrieval using the raw cluster items to capture niche evidence.
        try:
            if clusters:
                edge_threshold = 2
                for cid, items in clusters.items():
                    if len(items) <= edge_threshold:
                        # items are the original subquery strings
                        for item in items:
                            # perform a focused retrieval for this edge item
                            docs = self.retrieval_client.search(item, k=max(3, self.retrieval_k))
                            collected.extend(docs)
        except Exception:
            pass
        # (Note: primary retrieval executed above)
        dedup = {}
        for doc in collected:
            existing = dedup.get(doc.doc_id)
            if not existing or doc.score > existing.score:
                dedup[doc.doc_id] = doc
        top_docs = sorted(dedup.values(), key=lambda d: d.score, reverse=True)[: self.retrieval_k]
        return Evidence(agent_id="shared", top_docs=top_docs)

    def _collect_answers(self, question: Question, queries: List[AgentQuery], shared_evidence: Evidence) -> List[AgentAnswer]:
        answers: List[AgentAnswer] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for aq in queries:
                worker = SingleAgentWorker(
                    aq.agent_id,
                    self.models.get(aq.agent_id, ""),
                    self.retrieval_client,
                    self.api_config,
                    self.retrieval_k,
                )
                futures[executor.submit(worker.answer_with_shared_evidence, question, aq, shared_evidence)] = aq.agent_id
            for future in as_completed(futures):
                try:
                    answers.append(future.result())
                except Exception:
                    answers.append(
                        AgentAnswer(
                            agent_id=futures[future],
                            reasoning_chain="Agent failed during answer stage.",
                            answer="Error during generation",
                            uncertainty=1.0,
                            references=[],
                            evidence=shared_evidence,
                        )
                    )
        return answers
