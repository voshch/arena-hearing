from arena_hearing.constants import SEARCH_STATE, SPEED_LIMIT, detections, recorded_topics


def test_recorded_topics_cover_the_frontend_and_the_decision_record() -> None:
    rows = recorded_topics(("call",))
    topics = {row[0]: row[1] for row in rows}
    assert topics["hearing_call_detections"] == detections("{ns}", "call")
    assert topics["hearing_search_state"] == f"{{ns}}/{SEARCH_STATE}"
    assert topics["hearing_speed_limit"] == f"{{ns}}/{SPEED_LIMIT}"
    assert len(topics) == len(rows)
    assert all(row[3] and row[8] for row in rows)


def test_one_detection_row_per_frontend() -> None:
    keys = [row[0] for row in recorded_topics(("srp", "call")) if row[0].endswith("_detections")]
    assert keys == ["hearing_srp_detections", "hearing_call_detections"]


def test_latched_topics_are_recorded_transient_local() -> None:
    latched = {row[0] for row in recorded_topics(("call",)) if row[5]}
    assert latched == {"hearing_search_state", "hearing_slam_map"}
