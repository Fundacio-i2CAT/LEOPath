import pytest

from leopath.network_state.gsl_attachment.multihoming import (
    select_multihoming_attachments,
)


def test_independent_top_k_is_the_unconstrained_upper_bound() -> None:
    visibility = [
        [(10.0, 1), (20.0, 2)],
        [(11.0, 1), (30.0, 3)],
    ]

    selected, stats = select_multihoming_attachments(visibility, 1, "independent")

    assert selected == [[(10.0, 1)], [(11.0, 1)]]
    assert stats["gs_attachment_conflicts_unconstrained"] == 1.0


def test_exclusive_assignment_uses_each_satellite_once() -> None:
    visibility = [
        [(10.0, 1), (20.0, 2)],
        [(11.0, 1), (30.0, 3)],
    ]

    selected, stats = select_multihoming_attachments(visibility, 1, "exclusive")

    assert selected == [[(20.0, 2)], [(11.0, 1)]]
    assert len({sat for candidates in selected for _distance, sat in candidates}) == 2
    assert stats["gs_attachment_shortfall"] == 0.0


def test_exclusive_assignment_spreads_scarce_radios_before_second_links() -> None:
    visibility = [
        [(10.0, 1), (11.0, 2)],
        [(12.0, 1), (13.0, 2)],
    ]

    selected, stats = select_multihoming_attachments(visibility, 2, "exclusive")

    assert [len(candidates) for candidates in selected] == [1, 1]
    assert stats["gs_fully_attached"] == 0.0
    assert stats["gs_attachment_shortfall"] == 2.0


def test_assignment_preserves_extended_candidate_records() -> None:
    marker = object()
    selected, _stats = select_multihoming_attachments(
        [[(10.0, 1, marker), (20.0, 2, object())]],
        1,
        "exclusive",
    )

    assert selected[0][0] == (10.0, 1, marker)


@pytest.mark.parametrize("count", [0, -1])
def test_attachment_count_must_be_positive(count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        select_multihoming_attachments([], count)


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown gs_attachment_policy"):
        select_multihoming_attachments([], 1, "shared")


VISIBLE = [[(500.0, 1), (600.0, 2), (700.0, 3), (800.0, 4)]]  # satellites 2 and 4 northbound


def test_nearest_ascending_prefers_a_northbound_satellite() -> None:
    selected, _ = select_multihoming_attachments(
        VISIBLE, 1, order="nearest_ascending", ascending={2, 4}
    )
    assert [item[1] for item in selected[0]] == [2]


def test_one_per_half_gives_one_address_on_each_half() -> None:
    selected, _ = select_multihoming_attachments(VISIBLE, 2, order="one_per_half", ascending={2, 4})
    assert sorted(item[1] for item in selected[0]) == [1, 2]


def test_one_per_half_falls_back_to_length_when_a_half_is_empty() -> None:
    selected, _ = select_multihoming_attachments(VISIBLE, 2, order="one_per_half", ascending=set())
    assert sorted(item[1] for item in selected[0]) == [1, 2]


def test_direction_aware_orders_need_the_pass_directions() -> None:
    with pytest.raises(ValueError, match="pass directions"):
        select_multihoming_attachments(VISIBLE, 1, order="one_per_half")
