"""
pipeline/tools/datetime_tool.py  –  Current date/time + farming season tool.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict


def execute_get_current_datetime(timezone_offset_hours: float = 5.5) -> Dict[str, Any]:
    """Return current date/time in the given timezone (default IST)."""
    try:
        tz = timezone(timedelta(hours=timezone_offset_hours))
        now = datetime.now(tz)
        month = now.month
        # Simple agricultural season mapper for India
        if month in (2, 3, 4, 5):
            season = "Zaid / Spring (Zaid Maize season)"
        elif month in (6, 7, 8, 9, 10):
            season = "Kharif (Monsoon season)"
        else:
            season = "Rabi (Winter season)"
        return {
            "date":         now.strftime("%Y-%m-%d"),
            "day_of_week":  now.strftime("%A"),
            "time":         now.strftime("%H:%M:%S"),
            "timezone":     f"UTC{'+' if timezone_offset_hours >= 0 else ''}{timezone_offset_hours}",
            "farming_season": season,
            "datetime_iso": now.isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}
