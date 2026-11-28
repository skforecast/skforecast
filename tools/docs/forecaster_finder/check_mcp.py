"""
End to end check of the path "Using an AI assistant?" of the forecaster finder:
drives the MCP server of skforecast-ai as the client of an agent would
(profile, plan, create_cv, compare, get_code) with a daily series in a CSV
file, and fails if the comparison does not return a script.

It needs skforecast-ai with the extras `foundation` and `mcp`, which are not
dependencies of skforecast: run it in an environment that has them.

Usage:
    python -I check_mcp.py
"""
import asyncio
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def find(obj, key):
    """First value of `key` in a nested structure of dicts and lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            found = find(v, key)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = find(v, key)
            if found is not None:
                return found
    return None


async def main(root):
    index = pd.date_range("2018-01-01", periods=730, freq="D")
    rng = np.random.default_rng(123)
    t = np.arange(len(index))
    sales = 100 + 0.02 * t + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, len(index))
    os.makedirs(f"{root}/data")
    pd.DataFrame({"date": index, "sales": sales}).to_csv(f"{root}/data/my_data.csv", index=False)

    server = StdioServerParameters(
        command=os.path.join(os.path.dirname(sys.executable), "skforecast-ai"),
        args=["mcp", "--allow-dir", f"{root}/data", "--output-dir", f"{root}/out"],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("server:", init.server_info.name, init.server_info.version)
            print("tools:", [tool.name for tool in (await session.list_tools()).tools])

            async def call(name, **args):
                result = await session.call_tool(name, args)
                out = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                print(f"\n--- {name}({args}) error={result.is_error}\n{out[:600]}")
                try:
                    return json.loads(out)
                except ValueError:
                    return {"text": out}

            profile = await call("profile", data_path=f"{root}/data/my_data.csv",
                                 target="sales", date_column="date")
            plan = await call("plan", profile_id=profile["id"], steps=14)
            cv = await call("create_cv", plan_id=plan["id"])
            comparison = await call("compare", cv_id=cv["id"])
            best = find(comparison, "best_plan_id")
            code = await call("get_code", object_id=best) if best else {}
            return bool(best) and "skforecast" in json.dumps(code)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        ok = asyncio.run(main(folder))
    print("\nscript returned:", ok)
    sys.exit(0 if ok else 1)
