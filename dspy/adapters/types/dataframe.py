"""DataFrame type for DSPy signatures with RLM support.

This implementation uses Parquet serialization with PyArrow to preserve all pandas
data types when passing DataFrames to RLM sandbox environments.
"""

import io
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow  # noqa: F401
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def _is_dataframe(value: Any) -> bool:
    """Check if value is a pandas DataFrame without requiring pandas import."""
    type_module = getattr(type(value), "__module__", "")
    type_name = type(value).__name__
    return type_module.startswith("pandas") and type_name == "DataFrame"


class DataFrame(Type):
    """DataFrame type for DSPy signatures.

    Wraps pandas DataFrames for use in DSPy signatures. Supports auto-wrapping
    and attribute proxying for ergonomic usage.

    WARNING: dspy.DataFrame should only be used with dspy.RLM, which provides
    a Python sandbox where the DataFrame is available for code execution.
    Other modules (ChainOfThought, Predict, etc.) will only see a string
    representation of the DataFrame, not the actual data.

    Example:
        ```python
        class AnalyzeData(dspy.Signature):
            data: dspy.DataFrame = dspy.InputField()
            result: str = dspy.OutputField()

        # Pass pandas DataFrame directly (auto-wraps)
        rlm = dspy.RLM(AnalyzeData)
        result = rlm(data=my_pandas_dataframe)
        ```
    """

    data: Any  # pd.DataFrame at runtime

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, data: Any = None, **kwargs):
        """Create a DataFrame wrapper.

        Args:
            data: A pandas DataFrame or dspy.DataFrame instance

        Raises:
            TypeError: If data is not a pandas DataFrame
        """
        if data is not None and "data" not in kwargs:
            if _is_dataframe(data):
                kwargs["data"] = data
            elif isinstance(data, DataFrame):
                # Already wrapped - extract underlying DataFrame
                kwargs["data"] = data.data
            else:
                raise TypeError(
                    f"Expected pandas DataFrame, got {type(data).__name__}. "
                    f"Install pandas with: pip install pandas"
                )

        super().__init__(**kwargs)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_dataframe(cls, value: Any) -> Any:
        """Auto-wrap pandas DataFrames for Pydantic validation.

        This allows users to pass raw pandas DataFrames directly to signatures.
        """
        # If it's already a dict (from model_validate), pass through
        if isinstance(value, dict):
            return value
        # If it's a raw pandas DataFrame, wrap it
        if _is_dataframe(value):
            return {"data": value}
        # If it's already a DataFrame instance, extract the data field
        if isinstance(value, DataFrame):
            return {"data": value.data}
        return value

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying DataFrame.

        Allows df.shape, df.head(), etc. without accessing df.data explicitly.
        """
        if name.startswith("_") or name in ("data", "model_fields"):
            raise AttributeError(name)
        return getattr(self.data, name)

    def format(self) -> str:
        """Format for display purposes (returns string representation)."""
        return repr(self.data)

    @pydantic.model_serializer()
    def _serialize(self) -> list:
        """Serialize DataFrame to list of records for JSON serialization."""
        import warnings

        warnings.warn(
            "dspy.DataFrame is being serialized to JSON. "
            "Only dspy.RLM preserves DataFrame data; other modules receive a string representation.",
            UserWarning,
            stacklevel=6,
        )
        return self.data.to_dict(orient="records")

    # =========================================================================
    # RLM Sandbox Support
    # =========================================================================

    def sandbox_setup(self) -> str:
        """Return setup code for pandas in the sandbox."""
        return "import pandas as pd"

    def to_sandbox(self, var_name: str) -> tuple[str, bytes | None, str]:
        """Serialize DataFrame to Parquet for sandbox injection.

        Uses PyArrow Parquet format to preserve all pandas dtypes including:
        - int64, float64, bool
        - datetime64[ns] with timezone support
        - categorical data
        - nullable integer/boolean dtypes

        Args:
            var_name: Variable name in sandbox

        Returns:
            Tuple of (assignment_code, parquet_bytes, "parquet")
        """
        if not PYARROW_AVAILABLE:
            # Fall back to JSON serialization
            return None, None, None

        buffer = io.BytesIO()
        try:
            self.data.to_parquet(
                buffer,
                engine="pyarrow",
                index=True,
                compression="snappy"
            )
            parquet_bytes = buffer.getvalue()
        except Exception:
            # Fallback: convert problematic columns to string
            df_copy = self.data.copy()
            for col in df_copy.columns:
                try:
                    test_buffer = io.BytesIO()
                    df_copy[[col]].to_parquet(test_buffer, engine="pyarrow")
                except Exception:
                    df_copy[col] = df_copy[col].astype(str)

            buffer = io.BytesIO()
            df_copy.to_parquet(buffer, engine="pyarrow", index=True, compression="snappy")
            parquet_bytes = buffer.getvalue()

        assignment_code = f"{var_name} = pd.read_parquet('/tmp/dspy_vars/{var_name}.parquet')"
        return assignment_code, parquet_bytes, "parquet"

    @classmethod
    def from_sandbox(cls, data: Any) -> "DataFrame":
        """Reconstruct DataFrame from sandbox output."""
        if not PANDAS_AVAILABLE:
            return None

        if _is_dataframe(data):
            return cls(data=data)
        elif isinstance(data, (list, dict)):
            return cls(data=pd.DataFrame(data))
        return None

    def rlm_preview(self, max_chars: int = 500) -> str:
        """Generate LLM-friendly preview of DataFrame contents."""
        df = self.data
        lines = [
            f"DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns",
            "",
            "Columns:",
        ]

        for col in list(df.columns)[:10]:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            null_info = f" ({null_count:,} nulls)" if null_count > 0 else ""
            lines.append(f"  {col}: {dtype}{null_info}")

        if len(df.columns) > 10:
            lines.append(f"  ... and {len(df.columns) - 10} more columns")

        if len(df) > 0:
            lines.extend(["", "Sample (first 3 rows):", df.head(3).to_string()])

        preview = "\n".join(lines)
        return preview[:max_chars] + "..." if len(preview) > max_chars else preview
