from psiduck.aime import (
    extract_answer,
    extract_boxed,
    majority_vote,
    normalize_aime_answer,
)


def test_extract_boxed():
    assert extract_boxed(r"So the answer is \boxed{42}.") == 42
    assert extract_boxed(r"first \boxed{1} then \boxed{999}") == 999
    assert extract_boxed("no box here") is None


def test_normalize_range():
    assert normalize_aime_answer(1035) == 35
    assert normalize_aime_answer(999) == 999
    assert normalize_aime_answer(1000) == 0
    assert normalize_aime_answer(2016) == 16


def test_extract_answer_priority():
    # boxed wins over trailing integers
    assert extract_answer(r"maybe 7, but \boxed{116} total 20") == 116
    # answer label used when no box
    assert extract_answer("Final answer: 191 (done)") == 191
    # falls back to last integer
    assert extract_answer("we compute 12 then 35") == 35
    # nothing parseable
    assert extract_answer("no digits at all") is None
    assert extract_answer("") is None


def test_extract_answer_normalizes():
    assert extract_answer(r"\boxed{1035}") == 35


def test_majority_vote():
    assert majority_vote([1, 2, 2, 3]) == 2
    assert majority_vote([None, 5, None]) == 5
    assert majority_vote([None, None]) is None
    # tie broken by smallest
    assert majority_vote([9, 9, 4, 4]) == 4
