"""The one floating-point type of the library.

Every model, buffer, gradient, optimiser state, loss and metric uses
`DTYPE`. There is no autocast, no loss scaler and no float-to-float cast
anywhere, so no value is ever rounded to a different precision on its way
through the code and no rounding error can compound across conversions.

Setting the `UCSA_DTYPE` environment variable to `float32` (default) or
`float16` before import is the only change needed to run the whole library in
that type. `configure` makes it torch's default, so every tensor created
without an explicit dtype follows it. Integer and boolean tensors (token ids,
masks) are exact and may be converted to `DTYPE` with `to_dtype`.
"""

import os

import torch

SUPPORTED = {"float32": torch.float32, "float16": torch.float16}
ENVIRONMENT_VARIABLE = "UCSA_DTYPE"

DTYPE = SUPPORTED[os.environ.get(ENVIRONMENT_VARIABLE, "float32")]

# AdamW's epsilon must be representable: 1e-8 underflows to zero in float16.
ADAM_EPS = 1e-8 if torch.float32 == DTYPE else 1e-4


def configure() -> None:
    """Makes `DTYPE` the default dtype for newly created tensors."""
    torch.set_default_dtype(DTYPE)


def to_dtype(exact: torch.Tensor) -> torch.Tensor:
    """Converts an integer or boolean tensor to `DTYPE`, exactly.

    Args:
      exact: A tensor whose values are exactly representable (a mask, a
        count, an index).

    Returns:
      The same values as `DTYPE`.

    Raises:
      TypeError: If `exact` is already floating point, since converting
        between floating types is exactly what the library avoids.
    """
    if exact.is_floating_point():
        raise TypeError("to_dtype is for integer and boolean tensors only")
    return exact.to(DTYPE)
