from clamp import clamp


def test_clamp_keeps_value_inside_range() -> None:
    assert clamp(5, 1, 10) == 5


def test_clamp_uses_nearest_boundary() -> None:
    assert clamp(-1, 1, 10) == 1
    assert clamp(20, 1, 10) == 10

