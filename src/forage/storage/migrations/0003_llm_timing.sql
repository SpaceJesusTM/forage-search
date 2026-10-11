-- Per-call timing breakdown reported by the model server, for throughput and throttling analysis.
ALTER TABLE llm_call ADD COLUMN load_ms INTEGER NOT NULL DEFAULT 0;
ALTER TABLE llm_call ADD COLUMN prompt_eval_ms INTEGER NOT NULL DEFAULT 0;
ALTER TABLE llm_call ADD COLUMN eval_ms INTEGER NOT NULL DEFAULT 0;
