from vision.route_grouping import group_routes


def det(x, y, color, size=20):
    return {
        "bbox": [x - size / 2, y - size / 2, x + size / 2, y + size / 2],
        "color": {"name": color},
    }


def chain(color, start_x, start_y, n, step=80):
    return [det(start_x + i * step, start_y + i * step // 2, color) for i in range(n)]


def test_same_color_chain_is_one_route():
    dets = chain("red", 100, 100, 5)
    routes = group_routes(dets, 1000, 1000)
    assert len(routes) == 1
    assert len(routes[0]) == 5


def test_far_apart_same_color_splits():
    dets = chain("red", 50, 50, 3) + chain("red", 800, 800, 3)
    routes = group_routes(dets, 1000, 1000)
    assert len(routes) == 2


def test_colors_never_mix():
    dets = chain("red", 100, 100, 3) + chain("blue", 110, 110, 3)
    routes = group_routes(dets, 1000, 1000)
    assert len(routes) == 2
    for route in routes:
        assert len({d["color"]["name"] for d in route}) == 1


def test_min_grip_count_filters_small_groups():
    dets = chain("yellow", 100, 100, 2)
    routes = group_routes(dets, 1000, 1000)
    assert routes == []


def test_routes_sorted_by_size():
    dets = chain("red", 50, 50, 3) + chain("blue", 600, 100, 6)
    routes = group_routes(dets, 2000, 2000)
    assert [len(r) for r in routes] == [6, 3]


def test_similar_hue_bridges_palette_boundary():
    # ocker (wood) und gelb liegen im Farbton nah beieinander: eine Kette,
    # die die Palette in zwei Namen quantisiert, bleibt EINE Route
    def det_lab(x, y, name, hue_ab):
        d = det(x, y, name)
        import numpy as np
        a = 128 + 30 * np.cos(np.radians(hue_ab))
        b = 128 + 30 * np.sin(np.radians(hue_ab))
        d["color"]["lab"] = [140.0, float(a), float(b)]
        return d

    dets = [
        det_lab(100, 500, "wood", 70),
        det_lab(110, 420, "yellow", 84),
        det_lab(95, 340, "wood", 74),
        det_lab(105, 260, "yellow", 86),
    ]
    routes = group_routes(dets, 1000, 1000)
    assert len(routes) == 1
    assert len(routes[0]) == 4


def test_different_hue_still_separates():
    def det_lab(x, y, name, hue_ab):
        d = det(x, y, name)
        import numpy as np
        a = 128 + 40 * np.cos(np.radians(hue_ab))
        b = 128 + 40 * np.sin(np.radians(hue_ab))
        d["color"]["lab"] = [140.0, float(a), float(b)]
        return d

    # rot (33°) und blau (-72°) verschränkt: bleiben getrennte Routen
    dets = (
        [det_lab(100 + i * 80, 500 - i * 40, "red", 33) for i in range(3)]
        + [det_lab(110 + i * 80, 505 - i * 40, "blue", -72) for i in range(3)]
    )
    routes = group_routes(dets, 1000, 1000)
    assert len(routes) == 2


def test_class_name_fallback_for_old_payloads():
    dets = [
        {"bbox": [0, 0, 20, 20], "class_name": "red"},
        {"bbox": [40, 40, 60, 60], "class_name": "red"},
        {"bbox": [80, 80, 100, 100], "class_name": "red"},
    ]
    routes = group_routes(dets, 500, 500)
    assert len(routes) == 1
