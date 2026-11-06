"""
Registry resolution and loading errors of T0Adapter: an ONNX repo accepted by
the registry prefix and a repo that does not exist.
"""

from importlib.metadata import version

from skforecast.foundation._adapters import _resolve_adapter

print("tfc-t0", version("tfc-t0"), "| huggingface_hub", version("huggingface_hub"))
for model_id in [
    "theforecastingcompany/t0-alpha-onnx-int8",
    "theforecastingcompany/t0-does-not-exist",
]:
    adapter = _resolve_adapter(model_id)(model_id)
    print(model_id, "->", type(adapter).__name__)
    try:
        adapter._load_model()
        print("  loaded")
    except Exception as exc:
        print("  ", type(exc).__name__, ":", str(exc)[:160])
        print("   cause:", type(exc.__cause__).__name__, ":", str(exc.__cause__)[:200])
