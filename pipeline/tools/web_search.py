"""
pipeline/tools/web_search.py  –  Web search tool using DuckDuckGo + Google News RSS.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
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


def execute_web_search(
    query: str,
    max_results: int = 3,
    state: str = "Uttar Pradesh",
    country: str = "India",
) -> Dict[str, Any]:
    """Web search with robust public-source fallbacks (no API key required).

    Tries DuckDuckGo Instant Answer first, then Google News RSS.
    Results are geo-scoped to Uttar Pradesh / India.
    """
    raw_query     = (query or "").strip()
    state_scope   = (state or "Uttar Pradesh").strip()
    country_scope = (country or "India").strip()
    scoped_query  = raw_query

    lowered  = raw_query.lower()
    has_india = "india" in lowered
    has_up    = "uttar pradesh" in lowered or " u.p" in lowered or "up " in lowered
    if not (has_india or has_up):
        scoped_query = f"{raw_query} in {state_scope}, {country_scope}"

    def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for row in rows:
            key = (row.get("url") or "", row.get("title") or "")
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _from_duckduckgo() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        today    = datetime.now(timezone.utc)
        week_ago = today - timedelta(days=7)
        date_hint       = f"after:{week_ago.strftime('%Y-%m-%d')}"
        time_scoped_q   = f"{scoped_query} {date_hint}"

        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": time_scoped_q, "format": "json", "no_redirect": 1, "no_html": 1},
            headers=_HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()

        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            rows.append({
                "title":   data.get("Heading", "Result"),
                "snippet": abstract,
                "url":     data.get("AbstractURL", ""),
                "source":  "duckduckgo_instant",
            })

        for item in data.get("RelatedTopics") or []:
            if len(rows) >= max_results:
                break
            if isinstance(item, dict) and "Text" in item:
                rows.append({
                    "title":   "Related",
                    "snippet": item.get("Text", ""),
                    "url":     item.get("FirstURL", ""),
                    "source":  "duckduckgo_instant",
                })
        return rows

    def _from_google_news_rss() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rss_url = (
            "https://news.google.com/rss/search"
            f"?q={quote_plus(scoped_query)}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        r = requests.get(rss_url, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        root    = ET.fromstring(r.text)
        now     = datetime.now(timezone.utc)
        max_age = 7

        for item in root.findall("./channel/item"):
            if len(rows) >= max_results:
                break
            pub_date_str = item.findtext("pubDate")
            is_recent    = True
            if pub_date_str:
                try:
                    pub_date  = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                    pub_date  = pub_date.replace(tzinfo=timezone.utc)
                    is_recent = (now - pub_date).days <= max_age
                except Exception:
                    is_recent = True
            if not is_recent:
                continue
            title    = (item.findtext("title") or "").strip()
            link     = (item.findtext("link")  or "").strip()
            desc_raw = item.findtext("description") or ""
            desc     = unescape(re.sub(r"<[^>]+>", "", desc_raw)).strip()
            rows.append({
                "title":   title or "News result",
                "snippet": desc[:320],
                "url":     link,
                "source":  "google_news_rss",
                "pubDate": pub_date_str or "",
            })
        return rows

    try:
        providers = [_from_duckduckgo, _from_google_news_rss]
        merged: List[Dict[str, Any]] = []
        errors: List[str] = []

        for provider in providers:
            try:
                merged.extend(provider())
            except Exception as src_e:
                errors.append(f"{provider.__name__}: {src_e}")
            merged = _dedupe(merged)
            if len(merged) >= max_results:
                break

        # Prefer results mentioning the target state/country
        scope_tokens = [state_scope.lower(), "uttar pradesh", "india"]
        scoped_rows  = [
            row for row in merged
            if any(tok in f"{row.get('title','')} {row.get('snippet','')}".lower() for tok in scope_tokens)
        ]
        final_rows = scoped_rows if scoped_rows else merged

        payload: Dict[str, Any] = {
            "query":        raw_query,
            "scoped_query": scoped_query,
            "scope":        {"state": state_scope, "country": country_scope},
            "results":      final_rows[:max_results],
        }
        if errors and not payload["results"]:
            payload["error"] = "; ".join(errors)
        return payload

    except Exception as e:
        return {"error": str(e), "results": []}
