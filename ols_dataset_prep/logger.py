"""
Logger (STUB — Phase 1)

In Phase 1: all logging goes to console via Python's standard logging.
Langfuse integration is a no-op stub — functions exist but do nothing.

Phase 4 will replace the stub bodies with real Langfuse traces so every
pipeline run is tracked, every LLM call is logged, and every dataset
processed has a full audit trail at LANGFUSE_HOST.
"""

import logging
import os
from typing import Any, Optional


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean console handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class PipelineLogger:
    """
    Wraps Langfuse tracing. In Phase 1 all methods log to console only.
    In Phase 4 each method creates/updates a Langfuse trace.
    """

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.logger = logging.getLogger(f"ols.pipeline.{dataset_id}")
        self._trace_id: Optional[str] = None

        # Phase 4: initialize Langfuse client here
        # from langfuse import Langfuse
        # self._client = Langfuse(
        #     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        #     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        #     host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        # )

    def start_run(self, metadata: Optional[dict[str, Any]] = None) -> None:
        """[STUB] Start a pipeline run trace."""
        self.logger.info(f"Pipeline run started — {self.dataset_id}")
        # Phase 4: self._trace = self._client.trace(name=self.dataset_id, metadata=metadata)

    def log_stage(self, stage: str, data: Optional[dict[str, Any]] = None) -> None:
        """[STUB] Log a pipeline stage event."""
        self.logger.debug(f"Stage: {stage} — {data or ''}")
        # Phase 4: self._trace.span(name=stage, input=data)

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_used: int = 0,
    ) -> None:
        """[STUB] Log an LLM call for cost and quality tracking."""
        self.logger.debug(f"LLM call: {model} — {tokens_used} tokens")
        # Phase 4: self._trace.generation(model=model, input=prompt, output=response)

    def finish_run(self, success: bool, summary: Optional[dict[str, Any]] = None) -> None:
        """[STUB] Close the pipeline run trace."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Pipeline run {status} — {self.dataset_id} — {summary or ''}")
        # Phase 4: self._trace.update(output=summary, status_message=status)
