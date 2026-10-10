"""15일 스프린트 초안과 컴퓨트 예산 시각화 (기획서 v3 확정 전 초안)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

out = sys.argv[1] if len(sys.argv) > 1 else "sprint_plan.png"
BG, INK, MUTED, GRID = "#0f1115", "#e8e6e3", "#9aa0a6", "#2a2e36"
C = {"infra": "#7aa2f7", "data": "#9ece6a", "read": "#e0af68", "w1": "#bb9af7",
     "w2": "#f7768e", "w3": "#ff9e64", "solver": "#2ac3de", "duel": "#f5c2e7", "human": "#c0caf5"}

fig = plt.figure(figsize=(16, 12), facecolor=BG)
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.3], hspace=0.32, left=0.33, right=0.95, top=0.88, bottom=0.06)
fig.suptitle("Block Universe Cinema · 15일 스프린트 초안 (기획서 v3 확정 후 갱신)", color=INK, fontsize=21, x=0.03, ha="left", y=0.975)
fig.text(0.03, 0.935, "목표: Day 15에 같은 사진 한 장으로 Seedance I2V와 붙어서 눈으로 이기는 영상 · 소형 모델 Phase 1 전원 출전",
         color=MUTED, fontsize=12.5, ha="left")

# (a) 컴퓨트 예산
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(BG)
bars = [
    ("Kaggle 3계정 · 15일", 193, 386, C["infra"]),
    ("+ 상시 2×T4 임대 · 15일", 720, 720, C["solver"]),
    ("필요: Phase 1 스프린트판", 500, 775, C["w2"]),
    ("필요: + Phase 2 1세대", 786, 1128, C["w3"]),
]
for i, (lab, lo, hi, col) in enumerate(bars):
    ax.barh(i, lo, color=col, alpha=0.95, height=0.55)
    if hi > lo:
        ax.barh(i, hi - lo, left=lo, color=col, alpha=0.35, height=0.55, hatch="//", edgecolor=col)
    txt = f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"
    ax.text(hi + 25, i, txt + " T4-h", va="center", color=INK, fontsize=12)
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars], color=INK, fontsize=11.5)
ax.invert_yaxis(); ax.set_xlim(0, 1500)
ax.tick_params(axis="x", colors=MUTED); [s.set_visible(False) for s in ax.spines.values()]
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("컴퓨트 예산 (T4-h)", color=INK, fontsize=15, loc="left", pad=12)
ax.text(0, -0.30, "빗금 = 불확실 범위 (Kaggle T4×2 쿼터 차감률 1배/2배 미확인) · Kaggle만으로는 Phase 1 스프린트의 약 25–75%",
        color=MUTED, fontsize=10.5, va="top", ha="left", transform=ax.transAxes)

# (b) 간트
ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor(BG)
lanes = [
    ("인프라 · E0 교사 T4 구동·NaN 감사", [(0, 2, "infra")]),
    ("창업자 미션: IMU 산책·방 촬영·RAW 매크로", [(0, 8, "human")]),
    ("데이터 주조소: 렌더 GT·문법·견본·Realify", [(0, 7, "data"), (7, 13, "data")]),
    ("입력 판독 + 에셔 테스트", [(1, 4, "read")]),
    ("소형 1진: TAE-T·승강기·눈대중자·글자경찰\n감식반·리그감식·평론가·의도", [(2, 6, "w1")]),
    ("소형 2진: 독심헤드 MRH·레고설계사\n카메라 손·뒷면완성기", [(4, 9, "w2")]),
    ("소형 3진: TexelFM-35 · G-Adapter(E3 판정)", [(6, 12, "w3")]),
    ("세계 솔버 WSSC-T4 + 렌더·렌즈·센서 스택", [(3, 10, "solver"), (10, 14, "solver")]),
    ("같은 사진 대결 vs Seedance", [(9, 10, "duel"), (12, 13, "duel"), (14, 15, "duel")]),
]
for i, (lab, segs) in enumerate(lanes):
    for s, e, k in segs:
        ax2.barh(i, e - s, left=s, color=C[k], height=0.58, alpha=0.95)
ax2.set_yticks(range(len(lanes))); ax2.set_yticklabels([l[0] for l in lanes], color=INK, fontsize=11.5)
ax2.invert_yaxis(); ax2.set_xlim(0, 15); ax2.set_xticks(range(0, 16))
ax2.set_xticklabels([f"D{d}" for d in range(0, 16)], color=MUTED, fontsize=10)
[s.set_visible(False) for s in ax2.spines.values()]
ax2.grid(axis="x", color=GRID, lw=0.8); ax2.set_axisbelow(True)
for d, t in [(9.5, "1차 대결\n안전 무브"), (12.5, "2차 대결\n확장 무브"), (14.5, "최종 릴")]:
    ax2.annotate(t, xy=(d, 8.3), xytext=(d, 9.35), ha="center", color=C["duel"], fontsize=10,
                 arrowprops=dict(arrowstyle="-|>", color=C["duel"], lw=1.2))
ax2.set_ylim(9.9, -0.6)
ax2.set_title("15일 레인 (D0 = 기획서 v3 확정일)", color=INK, fontsize=15, loc="left", pad=12)
fig.savefig(out, dpi=110, facecolor=BG)
print(out)
