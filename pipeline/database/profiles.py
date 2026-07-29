import json
from typing import Any, Dict, Optional

from sqlalchemy import text

from pipeline.database.connection import get_async_db_session, run_async_db


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return list(parsed or [])
        except json.JSONDecodeError:
            return []
    return list(value or [])


def _as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return dict(parsed or {})
        except json.JSONDecodeError:
            return {}
    return dict(value or {})


async def upsert_user_profile_async(user_id: str, patch: Dict[str, Any]) -> None:
    """
    Merge-update the user_profiles row for user_id using the shared asyncpg pool.
    Only non-None values in `patch` are applied.
    """
    if not user_id or not patch:
        return

    try:
        async with get_async_db_session() as session:
            await session.execute(
                text("INSERT INTO user_profiles(user_id) VALUES (:user_id) ON CONFLICT DO NOTHING;"),
                {"user_id": user_id},
            )
            result = await session.execute(
                text("SELECT crops, extra_facts FROM user_profiles WHERE user_id = :user_id;"),
                {"user_id": user_id},
            )
            row = result.mappings().first() or {}
            existing_crops = _as_list(row.get("crops"))
            existing_extra = _as_dict(row.get("extra_facts"))

            new_crops = patch.get("crops")
            merged_crops = (
                list(dict.fromkeys(existing_crops + list(new_crops)))
                if new_crops
                else existing_crops
            )

            new_extra = patch.get("extra_facts")
            if new_extra:
                existing_extra.update(dict(new_extra))

            await session.execute(
                text(
                    """
                    UPDATE user_profiles SET
                        name            = COALESCE(:name, name),
                        language        = COALESCE(:language, language),
                        location        = COALESCE(:location, location),
                        state           = COALESCE(:state, state),
                        country         = COALESCE(:country, country),
                        sowing_date     = COALESCE(:sowing_date, sowing_date),
                        crop_stage      = COALESCE(:crop_stage, crop_stage),
                        latitude        = COALESCE(:latitude, latitude),
                        longitude       = COALESCE(:longitude, longitude),
                        farm_size_acres = COALESCE(:farm_size_acres, farm_size_acres),
                        soil_type       = COALESCE(:soil_type, soil_type),
                        crops           = CAST(:crops AS jsonb),
                        extra_facts     = CAST(:extra_facts AS jsonb),
                        updated_at      = now()
                    WHERE user_id = :user_id;
                    """
                ),
                {
                    "name": patch.get("name"),
                    "language": patch.get("language"),
                    "location": patch.get("location"),
                    "state": patch.get("state"),
                    "country": patch.get("country"),
                    "sowing_date": patch.get("sowing_date"),
                    "crop_stage": patch.get("crop_stage"),
                    "latitude": patch.get("latitude"),
                    "longitude": patch.get("longitude"),
                    "farm_size_acres": patch.get("farm_size_acres"),
                    "soil_type": patch.get("soil_type"),
                    "crops": json.dumps(merged_crops),
                    "extra_facts": json.dumps(existing_extra),
                    "user_id": user_id,
                },
            )
    except Exception as e:
        print(f"[DB] upsert_user_profile failed for {user_id}: {e}")


async def load_user_profile_async(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the full user profile dict, or None if not found."""
    if not user_id:
        return None

    try:
        async with get_async_db_session() as session:
            result = await session.execute(
                text(
                    """
                    SELECT user_id, name, language, location, state, country, sowing_date, crop_stage,
                           latitude, longitude, farm_size_acres, soil_type, crops, extra_facts, updated_at
                    FROM   user_profiles
                    WHERE  user_id = :user_id;
                    """
                ),
                {"user_id": user_id},
            )
            row = result.mappings().first()

        if row is None:
            return None

        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "language": row["language"],
            "location": row["location"],
            "state": row["state"],
            "country": row["country"],
            "sowing_date": row["sowing_date"],
            "crop_stage": row["crop_stage"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "farm_size_acres": row["farm_size_acres"],
            "soil_type": row["soil_type"],
            "crops": _as_list(row["crops"]),
            "extra_facts": _as_dict(row["extra_facts"]),
        }
    except Exception as e:
        print(f"[DB] load_user_profile failed for {user_id}: {e}")
        return None


def upsert_user_profile(user_id: str, patch: Dict[str, Any]) -> None:
    return run_async_db(upsert_user_profile_async(user_id, patch))


def load_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    return run_async_db(load_user_profile_async(user_id))
