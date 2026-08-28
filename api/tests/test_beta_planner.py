from vision.beta_planner import plan_beta


def det(x, y, size=20):
    return {"bbox": [x - size / 2, y - size / 2, x + size / 2, y + size / 2]}


def test_simple_ladder_alternates_hands():
    # senkrechte Leiter von unten (y groß) nach oben (y klein)
    route = [det(100, 500), det(110, 400), det(90, 300), det(105, 200)]
    steps = plan_beta(route)
    assert steps[0]["action"] == "start"
    assert steps[-1]["action"] == "top"
    assert steps[-1]["hand"] == "both"
    hands = [s["hand"] for s in steps if s["action"] == "move"]
    # Leiter ohne klare Seitlage: Hände wechseln sich ab
    assert all(a != b for a, b in zip(hands, hands[1:]))


def test_start_pair_uses_both_hands():
    route = [det(100, 500), det(160, 505), det(130, 380), det(120, 260)]
    steps = plan_beta(route)
    start_steps = [s for s in steps if s["action"] == "start"]
    assert len(start_steps) == 2
    assert {s["hand"] for s in start_steps} == {"left", "right"}
    assert start_steps[0]["step"] == start_steps[1]["step"]


def test_side_pull_uses_matching_hand():
    # deutlicher Rechtszug -> rechte Hand greift
    route = [det(100, 500), det(100, 400), det(220, 350), det(220, 250)]
    steps = plan_beta(route)
    side_step = [s for s in steps if s["hold"] == 2][0]
    assert side_step["hand"] == "right"


def test_long_move_inserts_match():
    # großer Abstand zwischen zweitem und drittem Griff -> vorher doppeln
    route = [det(100, 600), det(100, 520), det(100, 200), det(100, 120)]
    steps = plan_beta(route)
    assert any(s["action"] == "match" for s in steps)


def test_ordering_starts_at_lowest_hold():
    route = [det(100, 100), det(100, 500), det(100, 300)]
    steps = plan_beta(route)
    assert steps[0]["hold"] == 1  # y=500 ist am tiefsten


def test_single_hold_route_gives_no_beta():
    assert plan_beta([det(100, 100)]) == []
