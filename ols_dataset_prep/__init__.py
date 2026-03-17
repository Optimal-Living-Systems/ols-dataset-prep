"""
OLS Dataset Preparation Pipeline

Layer 2 of the OLS AI Lab stack. Fetches any HuggingFace dataset,
converts to snappy parquet, optionally augments with LLMs, validates,
and delivers to local disk and HuggingFace Hub for use in Unsloth Studio.
"""

__version__ = "0.1.0"
