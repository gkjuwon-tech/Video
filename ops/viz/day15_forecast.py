"""Day 15 예측 패널(빌더·회의론자·시청자 + 조정관) 결과: 같은 사진으로 Seedance와 블라인드 대결 시 우리 선호 확률."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
out = sys.argv[1]
BG, INK, MUTED = "#0f1115", "#e8e6e3", "#9aa0a6"
rows, cols = ["실사 사진", "사진형 AI\n(Soul 등)", "그림체 AI"], ["안전\n푸시인·슬라이더·±20°", "확장\n90°·워크스루", "풀\n360°·뒤돌아보기"]
S = {
    "P10 · 망한 판": np.array([[.24, .09, .05], [.18, .06, .03], [.15, .04, .02]]),
    "P50 · 중간 판": np.array([[.40, .20, .12], [.31, .14, .07], [.25, .07, .04]]),
    "P90 · 잘 풀린 판": np.array([[.53, .33, .22], [.45, .25, .15], [.36, .13, .07]]),
}
cmap = LinearSegmentedColormap.from_list("duel", ["#c0392b", "#3a2f35", "#1f6f50", "#2ecc71"])
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=0.7)
fig, axes = plt.subplots(1, 3, figsize=(17, 6.8), facecolor=BG)
fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.15, wspace=0.10)
for k, (ax, (name, m)) in enumerate(zip(axes, S.items())):
    ax.imshow(m, cmap=cmap, norm=norm)
    for i in range(3):
        for j in range(3):
            v = m[i, j]
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", color=INK, fontsize=18, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(cols, color=MUTED, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(rows if k == 0 else [""] * 3, color=INK, fontsize=12)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f"{name}  (9칸 평균 {m.mean()*100:.0f}%)", color=INK, fontsize=14, pad=10)
fig.suptitle("Day 15 예측: 같은 사진으로 Seedance와 블라인드 대결하면 일반 시청자가 우리를 고를 확률", color=INK, fontsize=18.5, x=0.03, ha="left", y=0.965)
fig.text(0.03, 0.885, "예측가 3명(빌더·회의론자·시청자) 독립 예측 → 조정관 근거 가중 보정 · 50% = 동률 · 초록 = 우리 우세 · 빨강 = Seedance 우세 · T4 하드웨어 가정(5090 전환 전)",
         color=MUTED, fontsize=11, ha="left")
fig.text(0.03, 0.025, "P50에서도 상위 20% 데모 릴에는 일반 시청자 55–65%가 우리를 고르는 클립이 3–6편 (체리피킹, 벤치와 분리 표기) · 30초 이상 안전 무브는 +6–10%p",
         color=MUTED, fontsize=11, ha="left")
fig.savefig(out, dpi=110, facecolor=BG)
print(out)
