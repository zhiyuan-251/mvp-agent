Multi-Agent Game-Theoretic QA (scaffold)
=======================================

Quick start
-----------
- Ensure the HotpotQA corpus file exists (default: `data/hotpotqa_corpus.json`). The repo already includes a small corpus; adjust `SEARCH_BACKEND.index_path` in `config.yaml` if you want a different source.
- Run the pipeline:
```bash
python main.py
```
Enter a question at the prompt to see the winning agent answer and per-agent scores.

Configuration
-------------
- Models: set in `config.yaml` under `MODELS`.
- Retrieval: BM25 over the local corpus. Configure `SEARCH_BACKEND.index_path` and `max_docs`. `RETRIEVAL.k` controls how many docs are fetched per query.
- Parallelism: control worker threads via `PARALLELISM.agent_workers`.
- Logging: `OUTPUT` toggles saving traces, agent reasoning, and evidence to `logs/<question-id>.json`.
- API calls: defaults point to SiliconFlow (`API.base_url`) using `DEEPSEEK_API_KEY`. Set `api_key_env` to `OPENAI_API_KEY` if you swap in OpenAI models, or override `base_url` per deployment.
- Partial decentralization: enabled by default (`PARTIAL_DECENTRALIZATION.enabled`). Agents decompose locally, sub-queries are clustered (k set by `cluster_k`), unified retrieval runs once, and all agents answer from shared evidence for consistency and lower cost.

Notes
-----
- The current agent reasoning/retrieval logic is lightweight and CPU-only for offline prototyping; swap in real model calls and a production BM25/embedding retriever as needed.
- The scoring module follows the provided game-theoretic weighting and includes a simple agreement metric; customize for your evaluation setup.
