"""UCSA-R: a causal, windowed language model with a persistent state.

Every logit is an exact next-token prediction, so the model is evaluated by
compression (bits per byte) and doubles as a lossless compressor. See
`ucsa.models.recurrent` for the model and `ucsa.training.engine` for training.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
