from psiduck.pipeline import LatentDelegationSolver, SingleModelSolver
from psiduck.sim_backbone import SimulatedBackbone, _try_exact_arithmetic


def test_exact_arithmetic_shapes_are_solved_precisely():
    assert _try_exact_arithmetic("What is 1 + 2 + 3 + ... + 10? Give the integer.") == 55
    assert _try_exact_arithmetic("Find the remainder when 7^100 is divided by 1000.") == 1
    assert (
        _try_exact_arithmetic(
            "The sum of the first n positive odd integers is 225. What is n?"
        )
        == 15
    )
    assert (
        _try_exact_arithmetic(
            "Let S be the sum of the positive divisors of 2023. Find the "
            "remainder when S is divided by 1000."
        )
        == 456
    )
    assert _try_exact_arithmetic("An unrelated word problem about geometry.") is None


def test_simulated_backbone_solves_parseable_problems_deterministically():
    bb = SimulatedBackbone(seed=1, realtime=False)
    problem = "Find the remainder when 7^100 is divided by 1000."
    bb.set_problem(problem, expected=1)

    baseline = SingleModelSolver(bb, max_new_tokens=64).solve(problem)
    latent = LatentDelegationSolver(bb, answer_max_new_tokens=8, num_samples=3).solve(
        problem
    )

    assert baseline.answer == 1
    assert latent.answer == 1
    # Same headline shape as the real backbone: latent generates far fewer tokens.
    assert latent.compute.generated_tokens < baseline.compute.generated_tokens


def test_simulated_backbone_never_reports_realtime_delay_when_disabled():
    bb = SimulatedBackbone(seed=2, realtime=False)
    bb.set_problem("some problem", expected=42)
    result = SingleModelSolver(bb, max_new_tokens=100).solve("some problem")
    assert result.wall_time_s < 0.05  # no sleeps happened


def test_latent_vector_is_deterministic_per_text():
    bb = SimulatedBackbone(seed=0, realtime=False)
    v1 = bb.latent_vector("hello world")
    v2 = bb.latent_vector("hello world")
    assert v1 == v2
    assert len(v1) == 8
