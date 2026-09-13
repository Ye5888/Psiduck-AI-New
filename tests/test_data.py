from psiduck.data import SAMPLE_PROBLEMS, get_problem, problems_by_difficulty


def test_problems_by_difficulty_filters():
    aime = problems_by_difficulty("aime")
    assert aime and all(p.difficulty == "aime" for p in aime)

    amc = problems_by_difficulty("amc")
    assert amc and all(p.difficulty == "amc" for p in amc)

    both = problems_by_difficulty("amc", "aime")
    assert len(both) == len(amc) + len(aime)

    everything = problems_by_difficulty()
    assert everything == list(SAMPLE_PROBLEMS)


def test_get_problem_by_id_and_index():
    p = get_problem("amc-style-odd-sum")
    assert p.answer == 15
    assert get_problem(0) is SAMPLE_PROBLEMS[0]


def test_sample_problems_have_unique_ids_and_valid_aime_range_answers():
    ids = [p.problem_id for p in SAMPLE_PROBLEMS]
    assert len(ids) == len(set(ids))
    for p in SAMPLE_PROBLEMS:
        assert 0 <= p.answer <= 999
