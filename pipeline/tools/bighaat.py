"""
pipeline/tools/bighaat.py  –  BigHaat product + blog search tool.

Uses Google News RSS as a reliable, no-auth search index for bighaat.com.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any, Dict, List
from urllib.parse import quote_plus

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def execute_bighaat_search(
    query: str,
    search_type: str = "both",   # "products" | "blogs" | "both"
    max_results: int = 3,
) -> Dict[str, Any]:
    """Search BigHaat for products and/or Kisan Vedika blog articles.

    Args:
        query:       Farmer's search term (English or Hindi).
        search_type: Which section to search – products, blogs, or both.
        max_results: Maximum results to return per category.

    Returns:
        dict with keys 'products', 'blogs', and 'query'.
    """
    results: Dict[str, Any] = {"products": [], "blogs": [], "query": query}

    def _fetch_from_google(scoped_query: str, limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            rss_url = (
                "https://news.google.com/rss/search"
                f"?q={quote_plus(scoped_query)}&hl=en-IN&gl=IN&ceid=IN:en"
            )
            r = requests.get(rss_url, headers=_HEADERS, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.text)

            for item in root.findall("./channel/item")[:limit]:
                title    = (item.findtext("title") or "").split(" - BigHaat")[0].strip()
                link     = (item.findtext("link")  or "").strip()
                desc_raw = item.findtext("description") or ""
                desc     = unescape(re.sub(r"<[^>]+>", "", desc_raw)).strip()

                # BigHaat often puts price in title: "Starting @ ₹1,120/-"
                price_match = re.search(r"₹\s*([\d,]+)", title)
                price = f"₹{price_match.group(1)}" if price_match else "See site"

                rows.append({
                    "title":   title,
                    "price":   price,
                    "snippet": desc[:250],
                    "url":     link,
                })
        except Exception:
            pass
        return rows

    # ── 1. Product search ──────────────────────────────────────────────────
    if search_type in ("products", "both"):
        rows = _fetch_from_google(f"site:bighaat.com/products {query}", max_results)
        for row in rows:
            row["source"] = "bighaat_product"
            results["products"].append(row)

    # ── 2. Kisan Vedika blog search ────────────────────────────────────────
    if search_type in ("blogs", "both"):
        rows = _fetch_from_google(f"site:bighaat.com/kisan-vedika {query}", max_results)
        for row in rows:
            row["source"] = "bighaat_blog"
            results["blogs"].append(row)

    return results
