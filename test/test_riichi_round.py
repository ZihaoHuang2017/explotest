from riichi_round_calc.riichi_types import (
    Wind,
    TransactionType,
    NewRound,
    Hand,
    ConcludedRound,
)
from riichi_round_calc.riichi_round import RiichiRound, generate_overall_score_deltas
from riichi_round_calc.round_end import generate_next_round, is_game_end
from dacite import from_dict


def test_normal_deal_in():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand = Hand(fu=30, han=1)
    riichi_round.add_deal_in(2, 0, hand)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": {"fu": 30, "han": 1},
                "score_deltas": [-1000, 0, 1000, 0],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-1000, 0, 1000, 0]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_normal_deal_in_30fu_1han_2_to_0():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand = Hand(fu=30, han=1)
    riichi_round.add_deal_in(0, 2, hand)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": {"fu": 30, "han": 1},
                "score_deltas": [1500, 0, -1500, 0],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [1500, 0, -1500, 0]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_deal_in_30fu_2han_1_to_0_0_1_riichi():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 3,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand = Hand(fu=30, han=2)
    riichi_round.add_deal_in(0, 1, hand)
    riichi_round.set_riichis([0, 1])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 3,
        "start_riichi_stick_count": 0,
        "riichis": [0, 1],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": {"fu": 30, "han": 2},
                "score_deltas": [2900, -2900, 0, 0],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [3900, -3900, 0, 0]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 3,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_double_ron():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 3,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand1 = Hand(fu=30, han=5)
    riichi_round.add_deal_in(0, 2, hand1)
    hand2 = Hand(fu=30, han=6)
    riichi_round.add_deal_in(1, 2, hand2)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 3,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN,
                "score_deltas": [8000, 0, -8000, 0],
                "hand": {"fu": 30, "han": 5},
            },
            {
                "transaction_type": TransactionType.DEAL_IN,
                "score_deltas": [0, 12000, -12000, 0],
                "hand": {"fu": 30, "han": 6},
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [8000, 12000, -20000, 0]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_double_ron_with_honba_and_riichi():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand1 = Hand(fu=30, han=6)
    riichi_round.add_deal_in(2, 3, hand1)
    hand2 = Hand(fu=30, han=2)
    riichi_round.add_deal_in(1, 3, hand2)
    riichi_round.set_riichis([1, 2, 3])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
        "riichis": [1, 2, 3],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN,
                "score_deltas": [0, 0, 12000, -12000],
                "hand": {"fu": 30, "han": 6},
            },
            {
                "transaction_type": TransactionType.DEAL_IN,
                "score_deltas": [0, 2300, 0, -2300],
                "hand": {"fu": 30, "han": 2},
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [0, 4300, 11000, -15300]
    expected_next_round = {
        "round_wind": Wind.SOUTH,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_double_ron_with_honba_and_riichi_with_a_dealer_win():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand1 = Hand(fu=30, han=6)
    riichi_round.add_deal_in(3, 1, hand1)
    hand2 = Hand(fu=30, han=2)
    riichi_round.add_deal_in(2, 1, hand2)
    riichi_round.set_riichis([1, 2, 3])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
        "riichis": [1, 2, 3],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": {"fu": 30, "han": 6},
                "score_deltas": [0, -18000, 0, 18000],
            },
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": {"fu": 30, "han": 2},
                "score_deltas": [0, -2300, 2300, 0],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [0, -21300, 4300, 17000]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 2,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_order_agnostic_for_double_ron():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand2 = Hand(fu=30, han=2)
    riichi_round.add_deal_in(1, 3, hand2)
    hand1 = Hand(fu=30, han=6)
    riichi_round.add_deal_in(2, 3, hand1)
    riichi_round.set_riichis([1, 2, 3])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
        "riichis": [1, 2, 3],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": {"fu": 30, "han": 2},
                "score_deltas": [0, 2300, 0, -2300],
            },
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": {"fu": 30, "han": 6},
                "score_deltas": [0, 0, 12000, -12000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [0, 4300, 11000, -15300]
    expected_next_round = {
        "round_wind": Wind.SOUTH,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_tsumo_30fu_3han_by_dealer():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand = Hand(fu=30, han=3)
    riichi_round.add_self_draw(1, hand)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.SELF_DRAW,
                "hand": {"fu": 30, "han": 3},
                "score_deltas": [-2000, 6000, -2000, -2000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-2000, 6000, -2000, -2000]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_tsumo_30fu_3han_by_non_dealer():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    hand = Hand(fu=30, han=3)
    riichi_round.add_self_draw(3, hand)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.SELF_DRAW,
                "hand": {"fu": 30, "han": 3},
                "score_deltas": [-1000, -2000, -1000, 4000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-1000, -2000, -1000, 4000]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 3,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_draw_with_one_tenpai_dealer():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    riichi_round.set_tenpais([0])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [0],
        "end_riichi_stick_count": 0,
        "transactions": [],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [3000, -1000, -1000, -1000]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_draw_with_one_tenpai_non_dealer():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    riichi_round.set_tenpais([1])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [1],
        "end_riichi_stick_count": 0,
        "transactions": [],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-1000, 3000, -1000, -1000]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_draw_with_all_tenpai_riichi_by_0():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    riichi_round.set_riichis([0])
    riichi_round.set_tenpais([0, 1, 2, 3])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [0],
        "tenpais": [0, 1, 2, 3],
        "end_riichi_stick_count": 1,
        "transactions": [],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-1000, 0, 0, 0]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 1,
        "start_riichi_stick_count": 1,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_draw_with_no_tenpai_1_riichi_stick_on_table():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 0,
        "start_riichi_stick_count": 1,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    riichi_round.set_tenpais([])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 2,
        "honba": 0,
        "start_riichi_stick_count": 1,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 1,
        "transactions": [],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [0, 0, 0, 0]
    expected_next_round = {
        "round_wind": Wind.EAST,
        "round_number": 3,
        "honba": 1,
        "start_riichi_stick_count": 1,
    }
    assert generate_next_round(ending_result) == NewRound.from_dict(expected_next_round)


def test_advance_to_south_1_from_draw():
    initial_round = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    round_params = NewRound.from_dict(initial_round)
    riichi_round = RiichiRound(round_params)
    riichi_round.set_tenpais([])
    next_round = generate_next_round(riichi_round.conclude_round())
    expected_next_round = {
        "round_wind": Wind.SOUTH,
        "round_number": 1,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    assert next_round == NewRound.from_dict(expected_next_round)


def test_south_3_going_to_south_4_with_p3_win_1st_at_30000():
    initial_round_s3 = {
        "round_wind": Wind.SOUTH,
        "round_number": 3,
        "honba": 0,
        "start_riichi_stick_count": 1,
    }
    round_params_s3 = NewRound.from_dict(initial_round_s3)
    round_s3 = RiichiRound(round_params_s3)
    hand_4000 = Hand(fu=30, han=3)
    round_s3.add_self_draw(3, hand_4000)
    ending_result_s3 = round_s3.conclude_round()

    assert generate_overall_score_deltas(ending_result_s3) == [
        -1000,
        -1000,
        -2000,
        5000,
    ]

    round_s3_next = generate_next_round(ending_result_s3)

    expected_round_s3_next = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }

    assert round_s3_next == NewRound.from_dict(expected_round_s3_next)
    assert not is_game_end(round_s3_next, [ending_result_s3])


def test_south_4_hanchan_end_with_p0_win_at_30000():
    initial_round_s4 = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 0,
        "start_riichi_stick_count": 1,
    }
    round_params_s4 = NewRound.from_dict(initial_round_s4)
    round_s4 = RiichiRound(round_params_s4)
    hand_4000 = Hand(fu=30, han=3)
    round_s4.add_self_draw(0, hand_4000)
    ending_result_s4 = round_s4.conclude_round()

    assert generate_overall_score_deltas(ending_result_s4) == [
        5000,
        -1000,
        -1000,
        -2000,
    ]

    round_s4_next = generate_next_round(ending_result_s4)

    expected_round_s4_next = {
        "round_wind": Wind.WEST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }

    assert round_s4_next == NewRound.from_dict(expected_round_s4_next)
    assert is_game_end(round_s4_next, [ending_result_s4])


def test_south_4_hanchan_not_end_with_p0_win_but_no_one_at_30000():
    round_s4 = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 4,
                "honba": 0,
                "start_riichi_stick_count": 0,
            },
        )
    )
    hand = from_dict(Hand, {"fu": 30, "han": 1})
    round_s4.add_self_draw(0, hand)
    ending_result_s4 = round_s4.conclude_round()

    assert generate_overall_score_deltas(ending_result_s4) == [1100, -300, -300, -500]

    round_s4_next = generate_next_round(ending_result_s4)

    expected_round_s4_next = {
        "round_wind": Wind.WEST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }

    assert round_s4_next == NewRound.from_dict(expected_round_s4_next)
    assert not is_game_end(round_s4_next, [ending_result_s4])


def test_south_4_hanchan_end_with_p3_win_less_than_30000_and_1st():
    riichi_round = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 4,
                "honba": 0,
                "start_riichi_stick_count": 0,
            },
        )
    )
    hand = from_dict(Hand, {"fu": 30, "han": 1})
    riichi_round.add_deal_in(3, 2, hand)
    ending_result = riichi_round.conclude_round()

    assert generate_overall_score_deltas(ending_result) == [0, 0, -1500, 1500]

    next_round = generate_next_round(ending_result)

    expected_next_round = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }

    assert next_round == NewRound.from_dict(expected_next_round)
    assert not is_game_end(next_round, [ending_result])


def test_south_4_hanchan_end_with_p3_win_at_30000_and_1st():
    riichi_round = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 4,
                "honba": 0,
                "start_riichi_stick_count": 2,
            },
        )
    )
    hand = from_dict(Hand, {"fu": 30, "han": 2})
    riichi_round.add_self_draw(3, hand)
    ending_result = riichi_round.conclude_round()

    assert generate_overall_score_deltas(ending_result) == [-1000, -1000, -1000, 5000]

    next_round = generate_next_round(ending_result)

    expected_next_round = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }

    assert next_round == NewRound.from_dict(expected_next_round)
    assert is_game_end(next_round, [ending_result])


def test_south_4_hanchan_end_with_p3_tenpai_at_30000_and_1st():
    round_s3 = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 3,
                "honba": 0,
                "start_riichi_stick_count": 0,
            },
        )
    )
    hand = from_dict(Hand, {"fu": 30, "han": 2})
    round_s3.add_deal_in(3, 2, hand)
    ending_result_s3 = round_s3.conclude_round()

    assert generate_overall_score_deltas(ending_result_s3) == [0, 0, -2000, 2000]

    round_s4 = RiichiRound(generate_next_round(ending_result_s3))
    round_s4.set_tenpais([3])
    ending_result_s4 = round_s4.conclude_round()

    assert generate_overall_score_deltas(ending_result_s4) == [
        -1000,
        -1000,
        -1000,
        3000,
    ]

    round_s4_next = generate_next_round(ending_result_s4)

    expected_round_s4_next = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }

    assert round_s4_next == NewRound.from_dict(expected_round_s4_next)
    assert is_game_end(round_s4_next, [ending_result_s3, ending_result_s4])


def test_south_4_hanchan_does_not_end_from_p0_score_gt_p3_score_gt_30000():
    riichi_round = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 4,
                "honba": 0,
                "start_riichi_stick_count": 0,
            },
        )
    )
    hand1 = Hand(fu=30, han=3)
    hand2 = Hand(fu=30, han=5)
    riichi_round.add_deal_in(3, 2, hand1)
    riichi_round.add_deal_in(0, 2, hand2)
    ending_result = riichi_round.conclude_round()

    assert generate_overall_score_deltas(ending_result) == [8000, 0, -13800, 5800]

    next_round = generate_next_round(ending_result)

    expected_next_round = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }

    assert next_round == NewRound.from_dict(expected_next_round)
    assert not is_game_end(next_round, [ending_result])


def test_south_4_hanchan_does_not_end_from_p3_not_1st_by_position():
    riichi_round = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 4,
                "honba": 0,
                "start_riichi_stick_count": 0,
            },
        )
    )
    hand1 = Hand(fu=30, han=5)
    hand2 = Hand(fu=30, han=6)
    riichi_round.add_deal_in(3, 2, hand1)
    riichi_round.add_deal_in(0, 2, hand2)
    ending_result = riichi_round.conclude_round()

    assert generate_overall_score_deltas(ending_result) == [12000, 0, -24000, 12000]

    next_round = generate_next_round(ending_result)

    expected_next_round = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }

    assert next_round == NewRound.from_dict(expected_next_round)
    assert not is_game_end(next_round, [ending_result])


def test_south_4_hanchan_end_with_p3_no_ten_and_p0_at_30000():
    round_s3 = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 3,
                "honba": 0,
                "start_riichi_stick_count": 0,
            },
        )
    )
    hand_s3 = Hand(fu=30, han=3)
    round_s3.add_self_draw(0, hand_s3)
    ending_result_s3 = round_s3.conclude_round()

    assert generate_overall_score_deltas(ending_result_s3) == [
        4000,
        -1000,
        -2000,
        -1000,
    ]

    round_s3_next = generate_next_round(ending_result_s3)

    expected_round_s3_next = {
        "round_wind": Wind.SOUTH,
        "round_number": 4,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }

    assert round_s3_next == NewRound.from_dict(expected_round_s3_next)

    round_s4 = RiichiRound(round_s3_next)
    round_s4.set_tenpais([0, 1, 2])
    ending_result_s4 = round_s4.conclude_round()

    assert generate_overall_score_deltas(ending_result_s4) == [1000, 1000, 1000, -3000]

    round_s4_next = generate_next_round(ending_result_s4)

    expected_round_s4_next = {
        "round_wind": Wind.WEST,
        "round_number": 1,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }

    assert round_s4_next == NewRound.from_dict(expected_round_s4_next)
    assert is_game_end(round_s4_next, [ending_result_s3, ending_result_s4])


def test_south_4_hanchan_does_not_end_with_p3_no_ten_and_no_one_at_30000():
    riichi_round = RiichiRound(
        from_dict(
            NewRound,
            {
                "round_wind": Wind.SOUTH,
                "round_number": 4,
                "honba": 0,
                "start_riichi_stick_count": 0,
            },
        )
    )
    riichi_round.set_tenpais([0])
    ending_result = riichi_round.conclude_round()

    assert generate_overall_score_deltas(ending_result) == [3000, -1000, -1000, -1000]

    next_round = generate_next_round(ending_result)

    expected_next_round = {
        "round_wind": Wind.WEST,
        "round_number": 1,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }

    assert next_round == NewRound.from_dict(expected_next_round)
    assert not is_game_end(next_round, [ending_result])


def test_hanchan_end_next_round_north_no_one_30000():
    round_params = {
        "round_wind": Wind.WEST,
        "round_number": 4,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    hand = Hand(fu=30, han=1)
    riichi_round.add_deal_in(0, 1, hand)
    ending_result = riichi_round.conclude_round()

    expected_overall_score_delta = [1000, -1000, 0, 0]
    next_round = generate_next_round(ending_result)

    assert generate_overall_score_deltas(ending_result) == expected_overall_score_delta
    assert next_round == from_dict(
        NewRound,
        {
            "round_wind": Wind.NORTH,
            "round_number": 1,
            "honba": 0,
            "start_riichi_stick_count": 0,
        },
    )
    assert is_game_end(next_round, [ending_result])


def test_hanchan_not_end_p1_at_0():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.set_riichis([1])
    hand = Hand(fu=30, han=8)
    riichi_round.add_deal_in(3, 1, hand)
    ending_result = riichi_round.conclude_round()

    expected_overall_score_delta = [0, -25000, 0, 25000]
    next_round = generate_next_round(ending_result)

    assert generate_overall_score_deltas(ending_result) == expected_overall_score_delta
    assert next_round == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 4,
            "honba": 1,
            "start_riichi_stick_count": 0,
        },
    )
    assert not is_game_end(next_round, [ending_result])


def test_hanchan_end_p1_less_than_0():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.set_riichis([1])
    hand = Hand(fu=30, han=8)
    riichi_round.add_deal_in(3, 1, hand)
    ending_result = riichi_round.conclude_round()

    expected_overall_score_delta = [0, -25300, 0, 25300]
    next_round = generate_next_round(ending_result)

    assert generate_overall_score_deltas(ending_result) == expected_overall_score_delta
    assert next_round == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 4,
            "honba": 2,
            "start_riichi_stick_count": 0,
        },
    )
    assert is_game_end(next_round, [ending_result])


def test_non_dealer_nagashi_mangan():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.add_nagashi_mangan(2)
    riichi_round.set_tenpais([])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [-4000, -2000, 8000, -2000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-4000, -2000, 8000, -2000]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 2,
            "honba": 1,
            "start_riichi_stick_count": 0,
        },
    )


def test_dealer_nagashi_mangan():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.add_nagashi_mangan(0)
    riichi_round.set_tenpais([])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [12000, -4000, -4000, -4000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [12000, -4000, -4000, -4000]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 2,
            "honba": 1,
            "start_riichi_stick_count": 0,
        },
    )


def test_non_dealer_nagashi_mangan_dealer_tenpai():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.add_nagashi_mangan(2)
    riichi_round.set_tenpais([0])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [0],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [-4000, -2000, 8000, -2000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-4000, -2000, 8000, -2000]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 1,
            "honba": 1,
            "start_riichi_stick_count": 0,
        },
    )


def test_non_dealer_nagashi_mangan_previous_riichi_sticks():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 1,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.add_nagashi_mangan(3)
    riichi_round.set_tenpais([])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 1,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 1,
        "transactions": [
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [-4000, -2000, -2000, 8000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-4000, -2000, -2000, 8000]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 2,
            "honba": 1,
            "start_riichi_stick_count": 1,
        },
    )


def test_non_dealer_nagashi_mangan_p1_p2_riichi():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.set_riichis([1, 2])
    riichi_round.add_nagashi_mangan(3)
    riichi_round.set_tenpais([1, 2])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [1, 2],
        "tenpais": [1, 2],
        "end_riichi_stick_count": 2,
        "transactions": [
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [-4000, -2000, -2000, 8000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-4000, -3000, -3000, 8000]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 2,
            "honba": 1,
            "start_riichi_stick_count": 2,
        },
    )


def test_three_nagashi_mangan():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    riichi_round.add_nagashi_mangan(1)
    riichi_round.add_nagashi_mangan(2)
    riichi_round.add_nagashi_mangan(3)
    riichi_round.set_tenpais([])
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 1,
        "honba": 0,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [-4000, 8000, -2000, -2000],
            },
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [-4000, -2000, 8000, -2000],
            },
            {
                "transaction_type": TransactionType.NAGASHI_MANGAN,
                "score_deltas": [-4000, -2000, -2000, 8000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-12000, 4000, 4000, 4000]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 2,
            "honba": 1,
            "start_riichi_stick_count": 0,
        },
    )


def test_pao_deal_in():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    hand_pao = Hand(fu=40, han=13)
    riichi_round.add_deal_in_pao(3, 1, 0, hand_pao)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN_PAO,
                "hand": hand_pao,
                "pao_target": 0,
                "score_deltas": [-24000, -24300, 0, 48300],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)


def test_pao_tsumo_yakuman():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    hand_pao = Hand(fu=40, han=13)
    hand = Hand(fu=30, han=13)
    riichi_round.add_self_draw_pao(3, 0, hand_pao)
    riichi_round.add_self_draw(3, hand)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.SELF_DRAW_PAO,
                "hand": hand_pao,
                "pao_target": 0,
                "score_deltas": [-48000, 0, 0, 48000],
            },
            {
                "transaction_type": TransactionType.SELF_DRAW,
                "hand": hand,
                "score_deltas": [-16100, -16100, -16100, 48300],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [
        -64100,
        -16100,
        -16100,
        96300,
    ]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 4,
            "honba": 2,
            "start_riichi_stick_count": 0,
        },
    )


def test_pao_deal_in_two_yakuman():
    round_params = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
    }
    riichi_round = RiichiRound(NewRound.from_dict(round_params))
    hand_pao = Hand(fu=40, han=13)
    hand = Hand(fu=30, han=13)
    riichi_round.add_deal_in_pao(3, 1, 0, hand_pao)
    riichi_round.add_deal_in(3, 1, hand)
    ending_result = riichi_round.conclude_round()

    expected_ending_result = {
        "round_wind": Wind.EAST,
        "round_number": 4,
        "honba": 1,
        "start_riichi_stick_count": 0,
        "riichis": [],
        "tenpais": [],
        "end_riichi_stick_count": 0,
        "transactions": [
            {
                "transaction_type": TransactionType.DEAL_IN_PAO,
                "hand": hand_pao,
                "pao_target": 0,
                "score_deltas": [-24000, -24300, 0, 48300],
            },
            {
                "transaction_type": TransactionType.DEAL_IN,
                "hand": hand,
                "score_deltas": [0, -48000, 0, 48000],
            },
        ],
    }

    assert ending_result == ConcludedRound.from_dict(expected_ending_result)
    assert generate_overall_score_deltas(ending_result) == [-24000, -72300, 0, 96300]
    assert generate_next_round(ending_result) == from_dict(
        NewRound,
        {
            "round_wind": Wind.EAST,
            "round_number": 4,
            "honba": 2,
            "start_riichi_stick_count": 0,
        },
    )
