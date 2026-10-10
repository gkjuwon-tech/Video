"""상시 T4 사냥 1차 결과: 15일(2×T4, 720 T4-h) 총액 비교. 검증 단계 진행 중 값."""
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
FX = 1380
START = {"즉시": "#9ece6a", "쿼터 대기(수분~일)": "#e0af68", "쿼터 불확실": "#f7768e"}
rows = [  # (이름, 저가USD, 고가USD, 시작, 메모)
    ("Vast.ai 체코 2×T4 · 입찰형", 59, 59, "즉시", "밀리면 일시정지"),
    ("Vast.ai 체코 2×T4 · 온디맨드", 127, 127, "즉시", "전 세계 유일한 2×T4 매물 · 개인 호스트"),
    ("Azure 한국중부 T4×1 두 대 · 스팟", 142, 142, "쿼터 대기(수분~일)", "무료체험 불가 · 종량제 전환"),
    ("AWS 서울 g4dn 두 대 · 스팟", 145, 145, "쿼터 대기(수분~일)", "중단 빈도 10–15%"),
    ("Vast.ai 2×Titan RTX (T4 동세대 대체)", 174, 174, "즉시", "16GB 상한 걸고 T4 흉내"),
    ("GCP 서울 2×T4 한 VM · 스팟", 260, 260, "쿼터 불확실", "신규 GPU 쿼터 0에서 시작"),
    ("GCP 서울 2×T4 한 VM · 온디맨드", 411, 455, "쿼터 불확실", "$300 크레딧으로 대부분 상쇄 가능(추정)"),
    ("AWS·Azure 서울 T4×1 두 대 · 온디맨드", 475, 475, "쿼터 대기(수분~일)", "안정"),
    ("Modal 서버리스 T4", 714, 714, "즉시", "상시 VM 아님 · 비쌈"),
]
fig, ax = plt.subplots(figsize=(15, 8.2), facecolor=BG)
fig.subplots_adjust(left=0.30, right=0.97, top=0.84, bottom=0.18)
ax.set_facecolor(BG)
for i, (n, lo, hi, s, memo) in enumerate(rows):
    ax.barh(i, lo * FX / 10000, color=START[s], height=0.6)
    if hi > lo:
        ax.barh(i, (hi - lo) * FX / 10000, left=lo * FX / 10000, color=START[s], alpha=0.4, height=0.6, hatch="//")
    lab = f"{lo*FX/10000:.0f}만원" if lo == hi else f"{lo*FX/10000:.0f}–{hi*FX/10000:.0f}만원"
    ax.text(hi * FX / 10000 + 1.5, i, f"{lab}  ·  {memo}", va="center", color=INK, fontsize=11)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=11.5)
ax.invert_yaxis(); ax.set_xlim(0, 175)
ax.tick_params(axis="x", colors=MUTED); [s.set_visible(False) for s in ax.spines.values()]
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("15일 총액 (만원, 1 USD = 1,380원 가정, 부가세 별도)", color=MUTED, fontsize=11)
fig.suptitle("상시 T4 사냥 1차 결과 · 2×T4 × 15일 (720 T4-h)", color=INK, fontsize=19, x=0.03, ha="left", y=0.965)
fig.text(0.03, 0.905, "T4는 멸종위기종: RunPod·TensorDock·Lambda·Paperspace·Jarvislabs·Thunder·Salad·Hyperstack 전부 T4 없음 · 검증 단계 진행 중 값",
         color=MUTED, fontsize=11.5, ha="left")
for j, (k, c) in enumerate(START.items()):
    fig.patches.append(plt.Rectangle((0.30 + j * 0.17, 0.025), 0.012, 0.025, color=c, transform=fig.transFigure, figure=fig))
    fig.text(0.315 + j * 0.17, 0.037, k, color=INK, fontsize=11, va="center")
fig.savefig(out, dpi=110, facecolor=BG)
print(out)
