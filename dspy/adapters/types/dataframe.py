"""DataFrame type for DSPy signatures with RLM support.

This implementation uses orjson serialization for efficient transfer of
DataFrames to RLM sandbox environments without requiring additional dependencies.
"""

import io
from typing import Any

import pydantic
import orjson

from dspy.adapters.types.base_type import Type

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def _is_dataframe(value: Any) -> bool:
    """Check if value is a pandas DataFrame without requiring pandas import."""
    type_module = getattr(type(value), "__module__", "")
    type_name = type(value).__name__
    return type_module.startswith("pandas") and type_name == "DataFrame"


def _serialize_df_to_json(df: "pd.DataFrame") -> bytes:
    """Serialize DataFrame to JSON bytes using orjson.
    
    Handles pandas/numpy types that orjson doesn't natively support by
    converting to Python-native types first.
    """
    # Convert to records format with proper type handling
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            # Handle pandas NA/NaT/NaN
            if pd.isna(val):
                record[col] = None
            # Handle numpy types
            elif hasattr(val, 'item'):  # numpy scalar
                record[col] = val.item()
            # Handle Timestamp
            elif isinstance(val, pd.Timestamp):
                record[col] = val.isoformat()
            # Handle Timedelta
            elif isinstance(val, pd.Timedelta):
                record[col] = str(val)
            else:
                record[col] = val
        records.append(record)
    
    # Include dtype info for reconstruction
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # Include index if it's not a simple RangeIndex
    index_data = None
    if not isinstance(df.index, pd.RangeIndex):
        index_data = {
            'values': [v.isoformat() if isinstance(v, pd.Timestamp) else v for v in df.index.tolist()],
            'name': df.index.name
        }
    
    payload = {
        'records': records,
        'dtypes': dtypes,
        'index': index_data,
        'columns': list(df.columns)
    }
    
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)


def _deserialize_json_to_df(json_bytes: bytes) -> "pd.DataFrame":
    """Deserialize JSON bytes back to DataFrame with dtype restoration."""
    payload = orjson.loads(json_bytes)
    
    records = payload['records']
    dtypes = payload.get('dtypes', {})
    index_data = payload.get('index')
    columns = payload.get('columns', [])
    
    # Create DataFrame from records
    df = pd.DataFrame(records, columns=columns if columns else None)
    
    # Restore dtypes where possible
    for col, dtype_str in dtypes.items():
        if col not in df.columns:
            continue
        try:
            if 'datetime64' in dtype_str:
                df[col] = pd.to_datetime(df[col])
            elif 'category' in dtype_str:
                df[col] = df[col].astype('category')
            elif 'Int64' in dtype_str or 'Int32' in dtype_str:
                df[col] = df[col].astype(dtype_str)
            elif 'Float64' in dtype_str or 'Float32' in dtype_str:
                df[col] = df[col].astype(dtype_str)
            elif dtype_str == 'bool':
                df[col] = df[col].astype(bool)
        except (ValueError, TypeError):
            pass  # Keep as-is if conversion fails
    
    # Restore index if present
    if index_data:
        df.index = pd.Index(index_data['values'], name=index_data.get('name'))
    
    return df


class DataFrame(Type):
    """DataFrame type for DSPy signatures.

    Wraps pandas DataFrames for use in DSPy signatures. Uses orjson for
    efficient serialization to RLM sandbox environments.

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
        return "import pandas as pd\nimport json"

    def to_sandbox(self, var_name: str) -> tuple[str, bytes | None, str]:
        """Serialize DataFrame to JSON for sandbox injection.

        Uses orjson for fast serialization with dtype hints for reconstruction.
        Handles pandas/numpy types including:
        - int64, float64, bool
        - datetime64[ns] (serialized as ISO strings)
        - categorical data (dtype hint preserved)
        - nullable integer/boolean dtypes

        Args:
            var_name: Variable name in sandbox

        Returns:
            Tuple of (assignment_code, json_bytes, "json")
        """
        if not PANDAS_AVAILABLE:
            return None, None, None

        json_bytes = _serialize_df_to_json(self.data)

        # Generate code that reads and reconstructs the DataFrame
        # Use stdlib json (not orjson) since orjson is not available in Pyodide/WASM
        assignment_code = f'''{var_name} = pd.DataFrame(json.loads(open('/tmp/dspy_vars/{var_name}.json').read())['records'])'''

        return assignment_code, json_bytes, "json"

    @classmethod
    def from_sandbox(cls, data: Any) -> "DataFrame":
        """Reconstruct DataFrame from sandbox output."""
        if not PANDAS_AVAILABLE:
            return None

        if _is_dataframe(data):
            return cls(data=data)
        elif isinstance(data, list):
            return cls(data=pd.DataFrame(data))
        elif isinstance(data, dict):
            return cls(data=pd.DataFrame(data))
        elif isinstance(data, bytes):
            # Reconstruct from JSON bytes
            df = _deserialize_json_to_df(data)
            return cls(data=df)
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
