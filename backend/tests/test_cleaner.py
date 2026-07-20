from app.preprocessing.cleaner import clean_text


def test_clean_text_normalizes_spacing() -> None:
    raw_text = (
        "Section 1: Purpose\r\n"
        "\r\n"
        "\r\n"
        "Enhanced      review is required.   "
    )

    expected_text = (
        "Section 1: Purpose\n\n"
        "Enhanced review is required."
    )

    assert clean_text(raw_text) == expected_text


def test_clean_text_preserves_policy_values() -> None:
    raw_text = (
        "Transfers above $10,000 require "
        "enhanced review."
    )

    cleaned_text = clean_text(raw_text)

    assert "$10,000" in cleaned_text