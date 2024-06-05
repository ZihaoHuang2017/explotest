import pytest

from riichi_round_calc.points import calculate_hand_value
from riichi_round_calc.riichi_types import Hand


@pytest.mark.parametrize(
    "fu, han, multiplier, expected",
    [
        (40, 1, 6, 2000),
        (110, 2, 6, 10600),
        (40, 1, 2, 700),
        (110, 2, 2, 3600),
        (40, 1, 4, 1300),
        (110, 2, 4, 7100),
        (40, 1, 1, 400),
        (110, 2, 1, 1800),
        (40, 4, 6, 12000),
        (70, 3, 6, 12000),
        (40, 4, 4, 8000),
        (70, 3, 4, 8000),
        (20, 5, 2, 4000),
        (20, 5, 1, 2000),
        (20, 6, 2, 6000),
        (20, 7, 2, 6000),
        (20, 8, 2, 8000),
        (20, 9, 2, 8000),
        (20, 10, 2, 8000),
        (20, 11, 2, 12000),
        (20, 12, 2, 12000),
        (20, 13, 2, 16000),
    ],
)
def test_points(fu: int, han: int, multiplier: int, expected: int):
    # Multiplier = 1: Non-dealer tsumo; 2: Dealer tsumo; 4: nondealer deal-in, 6: dealer deal-in
    assert calculate_hand_value(multiplier, Hand(fu, han)) == expected
