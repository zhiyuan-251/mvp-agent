import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from utils.schemas import Document

# Simple module-level cache to avoid rebuilding the index repeatedly.
_INDEX_CACHE: Dict[str, "BM25Client"] = {}


class BM25Client:
    def __init__(self, index_path: Optional[str] = None, max_docs: int = 50000):
        self.index_path = index_path or self._default_corpus()
        self.max_docs = max_docs
        self.documents: List[Document] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.k1 = 1.5
        self.b = 0.75
        if self.index_path:
            self._load_corpus()

    def _default_corpus(self) -> Optional[str]:
        for candidate in (
            "data/hotpotqa_corpus.json",
            "data/hotpotqa_corpus_full.json",
            "data/hotpotqa_corpus_large.json",
        ):
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_corpus(self) -> None:
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return
        total_len = 0
        for idx, item in enumerate(raw):
            if idx >= self.max_docs:
                break
            text = self._extract_text(item)
            tokens = self._tokenize(text)
            if not tokens:
                continue
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            doc_len = sum(freq.values())
            self.doc_lens.append(doc_len)
            total_len += doc_len
            doc_id = str(item.get("id") if isinstance(item, dict) and "id" in item else idx)
            self.documents.append(Document(doc_id=doc_id, text=text, score=0.0))
        self.avgdl = total_len / len(self.documents) if self.documents else 0.0
        self._compute_idf()

    def _extract_text(self, item: object) -> str:
        if isinstance(item, dict):
            if "sentence" in item and item["sentence"]:
                return str(item["sentence"])
            if "text" in item and item["text"]:
                return str(item["text"])
        if isinstance(item, list):
            return " ".join(str(x) for x in item)
        return str(item)

    def _compute_idf(self) -> None:
        if not self.doc_freqs:
            return
        N = len(self.doc_freqs)
        df: Counter = Counter()
        for freq in self.doc_freqs:
            df.update(freq.keys())
        self.idf = {
            term: max(0.0, math.log((N - freq + 0.5) / (freq + 0.5)))
            for term, freq in df.items()
        }

    def search(self, query: str, k: int = 5) -> List[Document]:
        if not query or not self.documents:
            return self._fallback(query, k)
        tokens = self._tokenize(query)
        if not tokens:
            return self._fallback(query, k)
        scores: List[Tuple[int, float]] = []
        for i, freq in enumerate(self.doc_freqs):
            doc_len = self.doc_lens[i] if i < len(self.doc_lens) else sum(freq.values())
            score = self._score(tokens, freq, doc_len)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:k] if scores else []
        return [
            Document(doc_id=self.documents[idx].doc_id, text=self.documents[idx].text, score=s)
            for idx, s in top
        ] or self._fallback(query, k)

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]

    def _score(self, query_tokens: List[str], doc_freq: Counter, doc_len: int) -> float:
        score = 0.0
        for term in query_tokens:
            if term not in doc_freq:
                continue
            df = doc_freq[term]
            idf = self.idf.get(term, 0.0)
            denom = df + self.k1 * (1 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
            score += idf * ((df * (self.k1 + 1)) / denom)
        return score

    def _fallback(self, query: str, k: int) -> List[Document]:
        snippet = (query or "")[:200]
        return [Document(doc_id=f"synthetic-{i}", text=snippet, score=0.01) for i in range(k)]


def search(query: str, k: int = 5, index_path: Optional[str] = None, max_docs: int = 50000) -> List[Document]:
    cache_key = index_path or "default"
    if cache_key in _INDEX_CACHE:
        client = _INDEX_CACHE[cache_key]
    else:
        client = BM25Client(index_path=index_path, max_docs=max_docs)
        _INDEX_CACHE[cache_key] = client
    return client.search(query, k=k)
