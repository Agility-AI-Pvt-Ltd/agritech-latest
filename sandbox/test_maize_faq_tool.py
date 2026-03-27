from unittest.mock import patch

from pipeline.tools.maize_faq import execute_faq_search_by_crop_stage, execute_set_crop_stage


def test_set_crop_stage_from_sowing_date():
    with patch("pipeline.tools.maize_faq.datetime") as mock_datetime:
        mock_datetime.strptime.side_effect = __import__("datetime").datetime.strptime
        mock_datetime.now.return_value = __import__("datetime").datetime(2026, 3, 27)
        result = execute_set_crop_stage("2026-03-06")

    assert result.get("crop_stage") == "15-30 Days"
    assert result.get("days_since_sowing") == 21


def test_direct_lookup_uses_stage_tree_without_qdrant():
    result = execute_faq_search_by_crop_stage(
        query="What should be the sowing time for summer maize?",
        crop_stage="0 Days",
        qdrant_client=None,
    )

    assert result.get("lookup_mode") == "direct_lookup"
    assert result["entries"][0]["subtopic"] == "Seed & Sowing"
    assert "mid-February to March" in result["entries"][0]["recommendation"]
