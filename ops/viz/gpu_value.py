"""GPU 가성비: Phase 1(600–930 T4-h 환산, 여유 20% 포함)을 끝내는 비용과 벽시계. Vast.ai 2026-10-10 실시간 최저가(신뢰도 ≥0.98) 기준."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
out = sys.argv[1]
BG, INK, MUTED, GRID = "#0f1115", "#e8e6e3", "#9aa0a6", "#2a2e36"
FX, W_LO, W_HI = 1380, 600, 930
# 이름, 머신 $/h, GPU 수, T4 대비 배속(저, 고), RT코어, 색
rows = [
    ("2×T4 (체코)", 0.298, 2, 1.0, 1.0, True, "#7aa2f7"),
    ("2×RTX 4090", 0.722, 2, 4.8, 8.9, True, "#9ece6a"),
    ("2×RTX 5090", 0.826, 2, 6.9, 11.7, True, "#73daca"),
    ("1×RTX 5090 (한국 호스트)", 0.213, 1, 6.9, 11.7, True, "#2ac3de"),
    ("1×A100 40GB", 0.402, 1, 5.0, 9.0, False, "#e0af68"),
    ("1×L40S 48GB", 0.601, 1, 5.5, 9.5, True, "#bb9af7"),
    ("1×H100 SXM 80GB", 1.480, 1, 7.5, 14.8, False, "#f7768e"),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.6), facecolor=BG, sharey=True)
fig.subplots_adjust(left=0.17, right=0.97, top=0.80, bottom=0.12, wspace=0.08)
for ax in (a1, a2):
    ax.set_facecolor(BG); [s.set_visible(False) for s in ax.spines.values()]
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(colors=MUTED)
for i, (n, dph, g, slo, shi, rt, c) in enumerate(rows):
    per_gpu = dph / g
    c_lo, c_hi = W_LO * per_gpu / shi * FX / 1e4, W_HI * per_gpu / slo * FX / 1e4
    d_lo, d_hi = W_LO / (g * shi) / 24, W_HI / (g * slo) / 24
    a1.barh(i, c_hi - c_lo if c_hi > c_lo else 0.4, left=c_lo, color=c, height=0.55)
    a1.text(c_hi + 0.6, i, f"{c_lo:.0f}–{c_hi:.0f}만원", va="center", color=INK, fontsize=11)
    a2.barh(i, d_hi - d_lo if d_hi > d_lo else 0.15, left=d_lo, color=c, height=0.55)
    a2.text(d_hi + 0.3, i, f"{d_lo:.1f}–{d_hi:.1f}일" + ("" if rt else "  · RT코어 없음"), va="center", color=INK, fontsize=11)
a1.set_yticks(range(len(rows))); a1.set_yticklabels([r[0] for r in rows], color=INK, fontsize=12)
a1.invert_yaxis(); a1.set_xlim(0, 30); a2.set_xlim(0, 24)
a1.set_title("Phase 1 끝내는 데 드는 돈 (만원)", color=INK, fontsize=14, loc="left")
a2.set_title("Phase 1 끝내는 데 걸리는 벽시계 (일, 24시간 가동)", color=INK, fontsize=14, loc="left")
a2.axvline(15, color="#f7768e", lw=1.2, ls="--"); a2.text(15.2, 6.45, "스프린트 15일", color="#f7768e", fontsize=10.5)
fig.suptitle("T4 고집 vs 굵게 끝내기 · Vast.ai 실시간 최저가(신뢰도 ≥0.98) · 2026-10-10", color=INK, fontsize=19, x=0.03, ha="left", y=0.965)
fig.text(0.03, 0.885, "작업량 = Phase 1 장부 500–775 T4-h + 여유 20% · T4 대비 배속은 우리 작업 혼합(교사 추론 50% · 소형 모델 학습 35% · 렌더·CPU 15%) 추정치",
         color=MUTED, fontsize=11.5, ha="left")
fig.text(0.03, 0.04, "1 USD = 1,380원 · 디스크·저장소 몇천 원 별도 · 가격과 재고는 실시간으로 바뀜", color=MUTED, fontsize=10.5)
fig.savefig(out, dpi=110, facecolor=BG)
print(out)
