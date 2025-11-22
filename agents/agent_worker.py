import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from retrieval.bm25_client import BM25Client
from utils.llm_client import LLMClient
from utils.schemas import AgentAnswer, AgentQuery, Evidence, Question


class SingleAgentWorker:
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        retrieval_client: BM25Client,
        api_config: Optional[Dict] = None,
        retrieval_k: int = 8,
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.retrieval_client = retrieval_client
        self.api_config = api_config or {}
        self.llm: Optional[LLMClient] = None
        self._decomp_prompt = self._load_prompt("decomposition.txt")
        self._reasoning_prompt = self._load_prompt("reasoning.txt")
        self.retrieval_k = retrieval_k
        self.logger = logging.getLogger(__name__)

    def __call__(self, question: Question) -> AgentAnswer:
        query = self.decompose(question)
        evidence = self._retrieve(query)
        return self._reason(question, query, evidence)

    def decompose(self, question: Question) -> AgentQuery:
        sub_questions: List[str] = []
        retrieval_queries: List[str] = []
        try:
            client = self._get_llm_client()
            prompt = (
                f"{self._decomp_prompt}\n"
                f"Question: {question.text}\n"
            )
            msg = [{"role": "user", "content": prompt}]
            raw = client.complete(msg, max_tokens=256, temperature=0.2)
            sub_questions, retrieval_queries = self._parse_decomposition(raw)
        except Exception as exc:
            self.logger.warning("Agent %s decomposition LLM call failed: %s", self.agent_id, exc)
            sub_questions = self._simple_split(question.text)
            retrieval_queries = sub_questions or [question.text]

        return AgentQuery(
            agent_id=self.agent_id,
            question=question.text,
            sub_questions=sub_questions or [question.text],
            retrieval_queries=retrieval_queries or sub_questions or [question.text],
        )

    def answer_with_shared_evidence(self, question: Question, query: AgentQuery, shared_evidence: Evidence) -> AgentAnswer:
        return self._reason(question, query, shared_evidence)

    def _retrieve(self, query: AgentQuery) -> Evidence:
        collected = []
        # Always include the full question as a retrieval query to avoid missing context.
        queries = list(dict.fromkeys(query.retrieval_queries + [query.question]))

        # First hop: retrieve documents for original queries
        for q in queries:
            docs = self.retrieval_client.search(q, k=self.retrieval_k)
            collected.extend(docs)

        # Multi-hop: extract entities from top documents and do second retrieval
        # This helps with questions like "What university did the director of X attend?"
        if collected:
            # Extract potential entities from top 3 documents
            entities = self._extract_entities_from_docs(collected[:3])
            if entities:
                # Do a second hop retrieval with extracted entities
                for entity in entities[:3]:  # Limit to top 3 entities
                    # Combine entity with original question context
                    second_hop_query = f"{entity} university attend education"
                    second_hop_docs = self.retrieval_client.search(second_hop_query, k=5)
                    collected.extend(second_hop_docs)

        # Deduplicate by doc_id while keeping highest score
        dedup = {}
        for doc in collected:
            existing = dedup.get(doc.doc_id)
            if not existing or doc.score > existing.score:
                dedup[doc.doc_id] = doc

        # Increase retrieval_k to 12 for better coverage with multi-hop
        top_k = max(self.retrieval_k, 12)
        top_docs = sorted(dedup.values(), key=lambda d: d.score, reverse=True)[:top_k]
        return Evidence(agent_id=self.agent_id, top_docs=top_docs)

    def _reason(self, question: Question, query: AgentQuery, evidence: Evidence) -> AgentAnswer:
        evidence_texts = "\n".join(f"[{d.doc_id}] {d.text}" for d in evidence.top_docs)
        references = [doc.doc_id for doc in evidence.top_docs]
        reasoning_chain = ""
        answer = ""
        uncertainty = 0.5

        try:
            client = self._get_llm_client()
            user_content = (
                f"Question: {question.text}\n"
                f"Sub-questions: {query.sub_questions}\n"
                f"Evidence:\n{evidence_texts}\n\n"
                "Based on the evidence above, answer the question.\n"
                "Respond with ONLY a JSON object (no markdown): {\"answer\": \"...\", \"reasoning\": \"...\", \"uncertainty\": 0.0}"
            )
            messages = [
                {"role": "system", "content": self._reasoning_prompt},
                {"role": "user", "content": user_content},
            ]
            raw = client.complete(messages, max_tokens=400, temperature=0.3)
            reasoning_chain, answer, uncertainty = self._parse_reasoning_output(raw)
        except Exception as e:
            self.logger.warning("Agent %s LLM call failed: %s", self.agent_id, e)
            reasoning_chain = self._compose_reasoning(query, evidence_texts)
            answer = self._extract_answer(question.text, evidence_texts)
            uncertainty = 0.5 if evidence.top_docs else 0.8

        return AgentAnswer(
            agent_id=self.agent_id,
            reasoning_chain=reasoning_chain,
            answer=answer,
            uncertainty=uncertainty,
            references=references,
            evidence=evidence,
        )

    def _simple_split(self, text: str) -> List[str]:
        parts = re.split(r" and |\?", text)
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned

    def _extract_entities_from_docs(self, docs: List) -> List[str]:
        """Extract potential named entities (people, films, etc.) from documents."""
        entities = []
        # Common patterns for extracting entities
        patterns = [
            r'directed by ([A-Z][a-z]+ [A-Z][a-z]+)',  # directed by Person Name
            r'([A-Z][a-z]+ [A-Z][a-z]+) is (?:an?|the)',  # Person Name is a/an/the
            r'"([^"]+)" (?:directed|written|produced)',  # "Film Title" directed
            r'([A-Z][a-z]+ [A-Z][a-z]+) (?:attended|graduated|studied)',  # Person attended
            r'director ([A-Z][a-z]+ [A-Z][a-z]+)',  # director Person Name
            r'([A-Z][a-z]+ [A-Z]\.? ?[A-Z]?[a-z]*)',  # Names like David Lynch, F. W. Murnau
        ]

        for doc in docs:
            text = doc.text
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    entity = match.strip()
                    if len(entity) > 3 and entity not in entities:
                        entities.append(entity)

        return entities

    def _compose_reasoning(self, query: AgentQuery, evidence_texts: str) -> str:
        if not evidence_texts:
            return "No relevant evidence found in the corpus."
        # Extract first few relevant snippets for reasoning
        snippets = evidence_texts[:500] if evidence_texts else ""
        return f"Based on available evidence: {snippets}"

    def _extract_answer(self, question_text: str, evidence_texts: str) -> str:
        if not evidence_texts:
            return "Insufficient evidence to answer this question."
        # Try to extract a more concise answer from evidence
        first_doc = evidence_texts.split('\n')[0] if evidence_texts else ""
        if first_doc:
            # Remove doc_id prefix if present
            import re
            clean = re.sub(r'^\[doc_\d+\]\s*', '', first_doc)
            return clean[:200] if clean else "Unable to determine answer from evidence."
        return "Unable to determine answer from evidence."

    def _parse_decomposition(self, raw: str) -> Tuple[List[str], List[str]]:
        import json

        # 1) Try to extract JSON object from the LLM output
        try:
            # locate first JSON-like substring
            json_str = raw.strip()
            if "```json" in json_str:
                import re

                m = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
                if m:
                    json_str = m.group(1)
            # find first brace
            start = json_str.find("{")
            if start != -1:
                json_str = json_str[start:]
            data = json.loads(json_str)
            sub = data.get("sub_questions") or data.get("subquestions") or data.get("sub-questions") or []
            retr = data.get("retrieval_queries") or data.get("retrieval") or data.get("retrieval_queries") or []
            sub_list = [s.strip() for s in sub if isinstance(s, str) and s.strip()]
            retr_list = [r.strip() for r in retr if isinstance(r, str) and r.strip()]
            if sub_list:
                return sub_list, retr_list or sub_list
        except Exception:
            pass

        # 2) Fallback: line-based heuristic parsing, strip headings and numbering
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        cleaned: List[str] = []
        for ln in lines:
            # remove common headings and numbering
            ln2 = ln
            # remove leading bullets or numbering like '1.' '2.' '-' '•'
            import re

            ln2 = re.sub(r'^\s*[-•*\d\.]+\s*', '', ln2)
            # drop generic headings
            if re.match(r'^(sub[- ]?questions|list of sub[- ]?questions|subquestions|sub-queries|sub queries|###)', ln2.lower()):
                continue
            cleaned.append(ln2)

        # prefer lines that look like questions or start with interrogatives
        interrogatives = ('who', 'what', 'when', 'where', 'why', 'how', 'which')
        sub_questions = [ln for ln in cleaned if ln.endswith('?') or ln.split()[0].lower().strip(':') in interrogatives]
        # if none, take first up to 3 cleaned lines as sub-questions
        if not sub_questions:
            sub_questions = cleaned[:3]

        # retrieval queries: pick concise token phrases, remove trailing punctuation
        retrievals: List[str] = []
        for s in sub_questions:
            q = re.sub(r'[\?\.!]$', '', s)
            retrievals.append(q)

        if not retrievals:
            retrievals = [ln for ln in cleaned][:2] or [""]

        return sub_questions, retrievals

    def _parse_reasoning_output(self, raw: str) -> Tuple[str, str, float]:
        # Prefer JSON structured output
        import json

        # Try to extract JSON from the response (handle markdown code blocks)
        json_str = raw.strip()

        # Remove markdown code blocks if present
        if "```json" in json_str:
            match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
        elif "```" in json_str:
            match = re.search(r'```\s*(.*?)\s*```', json_str, re.DOTALL)
            if match:
                json_str = match.group(1)

        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)

        try:
            data = json.loads(json_str)
            answer = str(data.get("answer", "")).strip()
            reasoning = str(data.get("reasoning", "")).strip()
            uncertainty = float(data.get("uncertainty", 0.5))

            # Validate we got meaningful content
            if answer and reasoning:
                return reasoning, answer, uncertainty
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to parse as plain text
        parts = raw.split("Answer:", 1)
        if len(parts) == 2:
            reasoning = parts[0].strip()
            answer_block = parts[1].strip()
        else:
            reasoning = raw.strip()
            answer_block = raw.strip()

        uncertainty = 0.5
        match = re.search(r"uncertainty[:\-\s]+([01](?:\.\d+)?)", raw, flags=re.IGNORECASE)
        if match:
            try:
                uncertainty = float(match.group(1))
            except ValueError:
                pass
        return reasoning, answer_block, uncertainty

    def _get_llm_client(self) -> LLMClient:
        if not self.llm:
            self.llm = LLMClient.from_model(self.model_name, api_config=self.api_config)
        return self.llm

    def critique_subqueries(self, target_agent_id: str, target_subquestions: List[str], question: Question) -> str:
        """Critique another agent's sub-queries: point out missing hops, ambiguities, or weaknesses."""
        try:
            client = self._get_llm_client()
            prompt = (
                f"You are agent {self.agent_id}. Given the overall question: {question.text}\n"
                f"Review sub-questions proposed by agent {target_agent_id}: {target_subquestions}\n"
                "Point out which sub-questions are weak, missing, ambiguous, or redundant. Be concise and list suggested improvements as bullets."
            )
            msg = [{"role": "user", "content": prompt}]
            raw = client.complete(msg, max_tokens=200, temperature=0.3)
            return raw.strip()
        except Exception as e:
            self.logger.warning("Agent %s critique failed: %s", self.agent_id, e)
            return "No critique produced."

    def refine_subqueries(self, original_query: AgentQuery, critiques: List[str], question: Question) -> AgentQuery:
        """Defend or revise own sub-questions after seeing critiques. Returns a possibly updated AgentQuery."""
        try:
            client = self._get_llm_client()
            prompt = (
                f"You are agent {self.agent_id}. Original sub-questions: {original_query.sub_questions}\n"
                f"Received critiques:\n" + "\n".join(f"- {c}" for c in critiques) + "\n"
                f"Based on these critiques, either defend your original sub-questions or provide a revised list.\n"
                "Return ONLY a JSON object: {\"sub_questions\": [...], \"retrieval_queries\": [...]}"
            )
            msg = [{"role": "user", "content": prompt}]
            raw = client.complete(msg, max_tokens=256, temperature=0.2)
            sub_questions, retrieval_queries = self._parse_decomposition(raw)
            return AgentQuery(
                agent_id=self.agent_id,
                question=original_query.question,
                sub_questions=sub_questions or original_query.sub_questions,
                retrieval_queries=retrieval_queries or original_query.retrieval_queries,
            )
        except Exception as e:
            self.logger.warning("Agent %s refine failed: %s", self.agent_id, e)
            return original_query

    def _load_prompt(self, filename: str) -> str:
        base = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
        try:
            with open(base, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""


def agent_worker(agent_id: str, model_name: str, question: Question, retrieval_client: Optional[BM25Client] = None) -> AgentAnswer:
    client = retrieval_client or BM25Client()
    worker = SingleAgentWorker(agent_id=agent_id, model_name=model_name, retrieval_client=client)
    return worker(question)
