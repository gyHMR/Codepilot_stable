from user import display_name


def test_display_name_strips_whitespace() -> None:
    assert display_name("  Ada  ", "ada@example.com") == "Ada"


def test_display_name_falls_back_to_email_local_part() -> None:
    assert display_name("   ", "grace@example.com") == "grace"

