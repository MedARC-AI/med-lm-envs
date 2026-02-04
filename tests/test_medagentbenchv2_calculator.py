from environments.medagentbenchv2.medagentbenchv2.tools import calculator


def test_calculator_basic_math() -> None:
    assert calculator(expression="1 + 2 * 3") == "7"


def test_calculator_decimal() -> None:
    assert calculator(expression="Decimal('1.1') + Decimal('2.2')") == "3.3"


def test_calculator_error_on_empty() -> None:
    assert calculator(expression="") == "Error: Empty expression"


def test_calculator_error_on_invalid() -> None:
    assert calculator(expression="1 / 0").startswith("Error:")
