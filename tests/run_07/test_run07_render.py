"""RUN-07 그림: 막대·진행 막대의 실제 픽셀 위치, 결정론, 글꼴 없음, 원자 쓰기."""
import math

import pytest
from PIL import Image
from run07_helpers import NOW, item

approx = pytest.approx


@pytest.mark.parametrize("values,box,vmax", [
    ([1.0, 2.0, 4.0], (0, 0, 30, 100), None),
    ([0.0, 0.0], (10, 20, 50, 60), None),
    ([5.0, 2.5, 0.0, 1.0], (80, 150, 1200, 430), 10.0),
])
def test_bar_rects_geometry(hr, values, box, vmax):
    x0, y0, x1, y1 = box
    rects = hr.bar_rects(values, box, vmax)
    assert len(rects) == len(values)
    vm = vmax if vmax is not None else (max(values) or 1.0)
    s = (x1 - x0) / len(values)
    for i, (v, (xa, ya, xb, yb)) in enumerate(zip(values, rects)):
        assert yb == y1
        assert (y1 - ya) == round((y1 - y0) * v / vm)
        assert xa == round(x0 + i * s + 0.15 * s) and xb == round(x0 + (i + 1) * s - 0.15 * s)
        assert x0 <= xa <= xb <= x1


def test_bar_rects_tallest_fills_box_and_heights_proportional(hr):
    rects = hr.bar_rects([1.0, 2.0, 4.0], (0, 0, 30, 100))
    assert [r[3] - r[1] for r in rects] == [25, 50, 100]


@pytest.mark.parametrize("values", [[1.0, -0.1], [1.0, float("nan")], [float("inf")]])
def test_bar_rects_rejects_bad_values(hr, values):
    with pytest.raises(ValueError):
        hr.bar_rects(values, (0, 0, 100, 100))


def test_bar_rects_rejects_degenerate_box(hr):
    with pytest.raises(ValueError):
        hr.bar_rects([1.0], (10, 10, 10, 50))
    with pytest.raises(ValueError):
        hr.bar_rects([1.0], (10, 50, 40, 20))


def _render(hr, d, tmp_path, name="r.png"):
    p = hr.render_png(d, tmp_path / name)
    return p, Image.open(p).convert("RGB")


def test_png_format_and_return_path(hr, data, tmp_path):
    out = tmp_path / "sub" / "dir" / "r.png"
    p = hr.render_png(data, out)
    assert p == out and out.exists()
    with Image.open(out) as im:
        im.verify()
    with Image.open(out) as im:
        assert im.format == "PNG" and im.mode == "RGB" and im.size == hr.CANVAS
    assert [f.name for f in out.parent.iterdir()] == ["r.png"]      # 임시 파일 잔여 없음


def test_usd_bars_at_exact_positions(hr, data, tmp_path):
    _, im = _render(hr, data, tmp_path)
    vals = data.hourly["usd"]
    rects = hr.bar_rects(vals, hr.CHART_BOX)
    x0, y0, x1, y1 = hr.CHART_BOX
    assert im.getpixel((5, 5)) == hr.COLOR_BG
    tall = max(range(24), key=lambda i: vals[i])
    assert tall == 23
    for i, (xa, ya, xb, yb) in enumerate(rects):
        cx = (xa + xb) // 2
        if vals[i] > 0:
            assert yb - ya > 0
            assert im.getpixel((cx, (ya + yb) // 2)) == hr.COLOR_USD
            assert im.getpixel((cx, yb - 1)) == hr.COLOR_USD
            assert im.getpixel((cx, ya)) == hr.COLOR_USD
            if ya - 3 >= y0:
                assert im.getpixel((cx, ya - 3)) != hr.COLOR_USD
        else:
            assert im.getpixel((cx, y1 - 2)) != hr.COLOR_USD
    # 막대 높이 비: 22번 칸(1.0)은 23번 칸(5.5)의 1/5.5 (±1 px)
    h22 = rects[22][3] - rects[22][1]
    h23 = rects[23][3] - rects[23][1]
    assert h23 == y1 - y0
    assert h22 == approx(h23 / 5.5, abs=1.01)


def test_chart_box_contains_bar_color_only_in_bars(hr, data, tmp_path):
    _, im = _render(hr, data, tmp_path)
    x0, y0, x1, y1 = hr.CHART_BOX
    rects = hr.bar_rects(data.hourly["usd"], hr.CHART_BOX)
    inside = lambda x, y: any(xa <= x < xb and ya <= y < yb for xa, ya, xb, yb in rects)  # noqa: E731
    stray = [(x, y) for y in range(y0, y1, 3) for x in range(x0, x1, 3)
             if im.getpixel((x, y)) == hr.COLOR_USD and not inside(x, y)]
    assert stray == []


def test_progress_bar_fill_width(hr, data, tmp_path):
    _, im = _render(hr, data, tmp_path)
    x0, y0, x1, y1 = hr.PROGRESS_BOX
    fw = round((x1 - x0) * 3 / 7)
    ym = (y0 + y1) // 2
    assert im.getpixel((x0 + 2, ym)) == hr.COLOR_DONE
    assert im.getpixel((x0 + fw - 3, ym)) == hr.COLOR_DONE
    assert im.getpixel((x0 + fw + 3, ym)) == hr.COLOR_TRACK
    assert im.getpixel((x1 - 3, ym)) == hr.COLOR_TRACK
    assert im.getpixel((x0 - 3, ym)) != hr.COLOR_DONE and im.getpixel((x1 + 3, ym)) != hr.COLOR_DONE


@pytest.mark.parametrize("n_done,n_total", [(0, 4), (4, 4)])
def test_progress_bar_extremes(hr, tmp_path, n_done, n_total):
    its = [item(f"U{i}", "done" if i < n_done else "todo") for i in range(n_total)]
    d = hr.collect(now=NOW, items=its, ledger=None)
    _, im = _render(hr, d, tmp_path)
    x0, y0, x1, y1 = hr.PROGRESS_BOX
    ym = (y0 + y1) // 2
    want = hr.COLOR_DONE if n_done else hr.COLOR_TRACK
    assert im.getpixel((x0 + 3, ym)) == want and im.getpixel((x1 - 4, ym)) == want


def test_empty_backlog_progress_is_empty_track(hr, tmp_path):
    d = hr.collect(now=NOW, items=[], ledger=None)
    _, im = _render(hr, d, tmp_path)
    x0, y0, x1, y1 = hr.PROGRESS_BOX
    assert im.getpixel((x0 + 3, (y0 + y1) // 2)) == hr.COLOR_TRACK


def test_no_ledger_renders_without_usd_bars(hr, items, tmp_path):
    d = hr.collect(now=NOW, items=items, ledger=None)
    _, im = _render(hr, d, tmp_path)
    x0, y0, x1, y1 = hr.CHART_BOX
    assert im.getpixel((5, 5)) == hr.COLOR_BG
    assert all(im.getpixel((x, y)) != hr.COLOR_USD for y in range(y0, y1, 4) for x in range(x0, x1, 4))


def test_title_text_is_drawn_outside_boxes(hr, data, tmp_path):
    _, im = _render(hr, data, tmp_path)
    top = im.crop((0, 0, hr.CANVAS[0], hr.CHART_BOX[1]))
    ink = sum(n for n, c in top.getcolors(maxcolors=1 << 20) if c != hr.COLOR_BG)
    assert ink >= 200               # 배경 외 글자 픽셀이 실제로 그려졌다


def test_render_is_deterministic_bytes(hr, data, tmp_path):
    a = hr.render_png(data, tmp_path / "a.png").read_bytes()
    b = hr.render_png(data, tmp_path / "b.png").read_bytes()
    assert a == b


def test_render_changes_when_data_changes(hr, data, items, ledger, tmp_path):
    a = hr.render_png(data, tmp_path / "a.png").read_bytes()
    other = hr.collect(now=NOW, items=items[:-2], ledger=ledger)
    b = hr.render_png(other, tmp_path / "b.png").read_bytes()
    assert a != b


def test_render_survives_missing_fonts(hr, data, tmp_path, monkeypatch):
    monkeypatch.setattr(hr, "FONT_CANDIDATES", ("/nonexistent/font.ttf", "/also/missing.ttc"))
    p, im = _render(hr, data, tmp_path)
    assert im.size == hr.CANVAS
    rects = hr.bar_rects(data.hourly["usd"], hr.CHART_BOX)
    xa, ya, xb, yb = rects[23]
    assert im.getpixel(((xa + xb) // 2, (ya + yb) // 2)) == hr.COLOR_USD


def test_render_overwrites_existing(hr, data, tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"old")
    hr.render_png(data, p)
    assert Image.open(p).size == hr.CANVAS
    assert math.isfinite(len(p.read_bytes()))
