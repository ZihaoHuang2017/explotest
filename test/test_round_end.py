import pytest
from riichi_round_calc.riichi_round import add_honba
from riichi_round_calc.riichi_types import TransactionType, Transaction


@pytest.mark.parametrize(
    "transaction_type, hand, score_deltas, pao_target, expected_score_deltas",
    [
        (
            TransactionType.DEAL_IN,
            {"fu": 30, "han": 1},
            [-1000, 0, 1000, 0],
            None,
            [-1900, 0, 1900, 0],
        ),
        (
            TransactionType.SELF_DRAW,
            {"fu": 30, "han": 2},
            [3000, -1000, -1000, -1000],
            None,
            [3900, -1300, -1300, -1300],
        ),
        (
            TransactionType.NAGASHI_MANGAN,
            None,
            [-4000, 12000, -4000, -4000],
            None,
            [-4000, 12000, -4000, -4000],
        ),
        (
            TransactionType.DEAL_IN_PAO,
            None,
            [-16000, 32000, -16000, 0],
            2,
            [-16900, 32900, -16000, 0],
        ),
        (
            TransactionType.SELF_DRAW_PAO,
            None,
            [0, 32000, -32000, 0],
            2,
            [0, 32900, -32900, 0],
        ),
    ],
)
def test_add_honba(
    transaction_type, hand, score_deltas, pao_target, expected_score_deltas
):
    ron_transaction = Transaction(
        transaction_type=transaction_type,
        hand=hand,
        pao_target=pao_target,
        score_deltas=score_deltas,
    )
    result = add_honba(ron_transaction, 3)
    assert result.score_deltas == expected_score_deltas
