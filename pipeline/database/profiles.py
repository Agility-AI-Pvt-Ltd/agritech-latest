import json
from typing import Any, Dict, Optional
import psycopg2.extras
from pipeline.database.connection import get_db_cursor


def upsert_user_profile(user_id: str, patch: Dict[str, Any]) -> None:
    """
    Merge-update the user_profiles row for user_id.
    Only non-None values in `patch` are applied.
    Scalar fields: name, language, location, state, country, sowing_date, latitude, longitude,
                   farm_size_acres, soil_type
    List fields:   crops  (merged, deduplicated)
    Dict fields:   extra_facts (merged)
    """
    if not user_id or not patch:
        return

    try:
        with get_db_cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Ensure row exists
            cur.execute(
                "INSERT INTO user_profiles(user_id) VALUES (%s) ON CONFLICT DO NOTHING;",
                (user_id,)
            )
            # Fetch current values
            cur.execute(
                "SELECT crops, extra_facts FROM user_profiles WHERE user_id = %s;",
                (user_id,)
            )
            row = cur.fetchone()
            existing_crops = list(row["crops"] or [])
            existing_extra = dict(row["extra_facts"] or {})

            # Merge crops (list)
            new_crops = patch.get("crops")
            if new_crops:
                merged_crops = list(dict.fromkeys(existing_crops + list(new_crops)))
            else:
                merged_crops = existing_crops

            # Merge extra_facts (dict)
            new_extra = patch.get("extra_facts")
            if new_extra:
                existing_extra.update(new_extra)

            cur.execute(
                """
                UPDATE user_profiles SET
                    name            = COALESCE(%s, name),
                    language        = COALESCE(%s, language),
                    location        = COALESCE(%s, location),
                    state           = COALESCE(%s, state),
                    country         = COALESCE(%s, country),
                    sowing_date     = COALESCE(%s, sowing_date),
                    latitude        = COALESCE(%s, latitude),
                    longitude       = COALESCE(%s, longitude),
                    farm_size_acres = COALESCE(%s, farm_size_acres),
                    soil_type       = COALESCE(%s, soil_type),
                    crops           = %s::jsonb,
                    extra_facts     = %s::jsonb,
                    updated_at      = now()
                WHERE user_id = %s;
                """,
                (
                    patch.get("name"),
                    patch.get("language"),
                    patch.get("location"),
                    patch.get("state"),
                    patch.get("country"),
                    patch.get("sowing_date"),
                    patch.get("latitude"),
                    patch.get("longitude"),
                    patch.get("farm_size_acres"),
                    patch.get("soil_type"),
                    json.dumps(merged_crops),
                    json.dumps(existing_extra),
                    user_id,
                )
            )
    except Exception as e:
        print(f"[DB] upsert_user_profile failed for {user_id}: {e}")


def load_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the full user profile dict, or None if not found."""
    if not user_id:
        return None
        
    try:
        with get_db_cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_id, name, language, location, state, country, sowing_date, latitude, longitude,
                       farm_size_acres, soil_type, crops, extra_facts, updated_at
                FROM   user_profiles
                WHERE  user_id = %s;
                """,
                (user_id,)
            )
            row = cur.fetchone()
            
        if row is None:
            return None
            
        return {
            "user_id":         row["user_id"],
            "name":            row["name"],
            "language":        row["language"],
            "location":        row["location"],
            "state":           row["state"],
            "country":         row["country"],
            "sowing_date":     row["sowing_date"],
            "latitude":        row["latitude"],
            "longitude":       row["longitude"],
            "farm_size_acres": row["farm_size_acres"],
            "soil_type":       row["soil_type"],
            "crops":           list(row["crops"] or []),
            "extra_facts":     dict(row["extra_facts"] or {}),
        }
    except Exception as e:
        print(f"[DB] load_user_profile failed for {user_id}: {e}")
        return None
