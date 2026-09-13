from psiduck.pipeline import (
    LatentDelegationSolver,
    SingleModelSolver,
    compare,
)
from psiduck.roles import DELEGATE_ROLES
from tests.fake_backbone import FakeBackbone


def test_latent_solver_shares_one_backbone_and_wiring():
    bb = FakeBackbone()
    solver = LatentDelegationSolver(bb)
    assert solver.backbone is bb
    assert solver.num_agents == 1 + len(DELEGATE_ROLES) + 1 == 6

    result = solver.solve("Some problem")
    # one encode + one message per delegate
    assert len(bb.encode_calls) == 1
    assert bb.delegate_roles == list(DELEGATE_ROLES)
    assert result.detail["latent_messages"] == 4
    assert result.detail["shared_backbone_id"] == id(bb)
    assert result.answer == 116


def test_latent_generates_far_fewer_tokens_than_baseline():
    bb = FakeBackbone(baseline_tokens=200)
    latent = LatentDelegationSolver(bb, answer_max_new_tokens=24).solve("p")
    baseline = SingleModelSolver(bb, max_new_tokens=256).solve("p")

    # The whole thesis: latent communication decodes almost no English.
    assert latent.compute.generated_tokens <= 3
    assert baseline.compute.generated_tokens == 200
    assert latent.compute.generated_tokens < baseline.compute.generated_tokens
    # Latent uses more (cheap) forward passes: encode + 4 delegates.
    assert latent.compute.forward_passes == 5
    assert baseline.compute.forward_passes == 0


def test_compare_reports_both_and_ratio():
    bb = FakeBackbone(baseline_tokens=180, answer_text="191")
    cmp = compare(bb, "prob", expected=191, baseline_max_new_tokens=180)
    assert cmp.baseline.approach == "single-llm-english"
    assert cmp.latent.approach == "latent-4-agent"
    assert cmp.latent.answer == 191
    ratio = cmp.generated_token_ratio
    assert ratio is not None and ratio < 0.1  # latent << baseline


def test_self_consistency_majority_votes_across_samples():
    # Two decodes say 116, one says 42 -- majority vote should win with 116,
    # and generated_tokens should scale with num_samples (still << baseline).
    bb = FakeBackbone(answer_texts=["116", "42", "116"], baseline_tokens=200)
    result = LatentDelegationSolver(bb, answer_max_new_tokens=8, num_samples=3).solve(
        "p"
    )
    assert result.answer == 116
    assert result.detail["sampled_answers"] == [116, 42, 116]
    assert result.detail["num_samples"] == 3
    assert result.approach == "latent-4-agent+sc3"
    # 3 samples of <=3 tokens each = 9, still far below a 200-token baseline.
    assert result.compute.generated_tokens == 9


def test_on_step_callback_reports_each_agent():
    bb = FakeBackbone()
    seen: list[str] = []
    LatentDelegationSolver(bb).solve("p", on_step=seen.append)
    assert seen[0] == "encoder"
    assert seen[1:5] == list(DELEGATE_ROLES)
    assert seen[5] == "decoder"


def test_meter_isolates_per_solve():
    bb = FakeBackbone(baseline_tokens=50)
    first = SingleModelSolver(bb, max_new_tokens=50).solve("a")
    second = SingleModelSolver(bb, max_new_tokens=50).solve("b")
    # Each SolveResult reports only its own delta, not the cumulative meter.
    assert first.compute.generated_tokens == 50
    assert second.compute.generated_tokens == 50
