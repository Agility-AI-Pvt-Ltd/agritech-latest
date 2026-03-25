"""
Simple runtime test for `web_search` tool.

Run:
    /Users/krishnakumar/Downloads/merged/.venv/bin/python sandbox/test_web_search.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import execute_web_search


def run_web_search_test() -> int:
    query = "latest agriculture news in India"
    print("=" * 72)
    print("TEST: web_search tool")
    print("=" * 72)
    print(f"Query: {query}")

    result = execute_web_search(query=query, max_results=3)
    print("\nRaw Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("error"):
        print("\n[FAIL] web_search returned error:", result["error"])
        return 1

    rows = result.get("results", [])
    if not isinstance(rows, list):
        print("\n[FAIL] Invalid payload: `results` is not a list")
        return 1

    print(f"\n[PASS] web_search executed successfully. Results count: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_web_search_test())
