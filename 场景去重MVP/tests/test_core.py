from scene_dedup.core import l2_normalize, parse_time, pick_task, task_bounds, task_rows


def test_time_parsing_and_bounds():
    assert parse_time("01:30") == 90
    assert parse_time("01:02:03.5") == 3723.5
    assert task_bounds({"start_frame": 30, "end_frame": 90}, fps=30) == (1.0, 3.0)
    assert task_bounds({"start": "00:10", "end": "00:12"}) == (10.0, 12.0)


def test_nested_task_schema():
    payload = {"data": {"tasks": [{"task_id": "task_3", "start": 1, "end": 2}]}}
    assert len(task_rows(payload)) == 1
    assert pick_task(payload, 3)["start"] == 1


def test_normalization():
    values = l2_normalize([[3.0, 4.0]])
    assert abs(float(values[0, 0]) - 0.6) < 1e-6
