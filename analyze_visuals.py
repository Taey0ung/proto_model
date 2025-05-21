import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ✅ 사용자 설정
model_id = "V6PlusHybrid_20250516_1255"
visual_dir = f"checkpoints/{model_id}/visuals"
label_names = ["sad", "angry", "neutral", "happy"]
epochs = list(range(1, 14))  # ✅ 수정 가능

# ✅ 1. 감정별 Gate 평균 변화 시뮬레이션 데이터 (실제는 log에서 추출 가능)
gate_means_by_class = {
    "sad":     [0.42, 0.55, 0.68, 0.73, 0.72, 0.74, 0.78, 0.80, 0.82, 0.81, 0.83, 0.82, 0.81],
    "angry":   [0.45, 0.60, 0.66, 0.70, 0.74, 0.79, 0.81, 0.83, 0.85, 0.84, 0.86, 0.85, 0.84],
    "neutral": [0.80, 0.78, 0.76, 0.74, 0.72, 0.71, 0.70, 0.68, 0.66, 0.64, 0.63, 0.62, 0.61],
    "happy":   [0.60, 0.65, 0.70, 0.73, 0.76, 0.78, 0.79, 0.81, 0.82, 0.83, 0.84, 0.85, 0.84]
}

# ✅ 2. 감정별 Fusion Alpha (sigmoid(alpha_weights)) 변화 시뮬레이션
fusion_alpha = {
    "sad":     [0.25, 0.28, 0.29, 0.28, 0.29, 0.28, 0.29, 0.28, 0.29, 0.26, 0.26, 0.31, 0.27],
    "angry":   [0.26, 0.28, 0.27, 0.30, 0.27, 0.30, 0.24, 0.25, 0.29, 0.25, 0.29, 0.26, 0.27],
    "neutral": [0.22, 0.17, 0.19, 0.14, 0.19, 0.17, 0.19, 0.18, 0.17, 0.22, 0.21, 0.19, 0.20],
    "happy":   [0.24, 0.27, 0.24, 0.26, 0.24, 0.24, 0.27, 0.27, 0.25, 0.27, 0.24, 0.24, 0.25]
}

# ✅ 3. Gate Mean Trend Plot
plt.figure(figsize=(8, 5))
for emotion in label_names:
    plt.plot(epochs, gate_means_by_class[emotion], label=emotion)
plt.xlabel("Epoch")
plt.ylabel("Gate Weight Mean")
plt.title("Classwise Gate Weight Mean Trend")
plt.ylim(0, 1.0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(visual_dir, "gate_mean_trend.png"))
plt.close()

# ✅ 4. Fusion Alpha Trend Plot
plt.figure(figsize=(8, 5))
for emotion in label_names:
    plt.plot(epochs, fusion_alpha[emotion], label=emotion)
plt.xlabel("Epoch")
plt.ylabel("Fusion Alpha (sigmoid)")
plt.title("Classwise Fusion Alpha Trend (Cosine Preference ↑)")
plt.ylim(0, 1.0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(visual_dir, "fusion_alpha_trend.png"))
plt.close()

# ✅ 5. Confusion Matrix 비교 (Epoch 2 vs Best Epoch)
try:
    cm2 = Image.open(os.path.join(visual_dir, "cm_epoch2.png"))
    cm_best = Image.open(os.path.join(visual_dir, "cm_epoch13.png"))  # best가 13이라고 가정
    combined = Image.new("RGB", (cm2.width * 2, cm2.height))
    combined.paste(cm2, (0, 0))
    combined.paste(cm_best, (cm2.width, 0))
    combined.save(os.path.join(visual_dir, "cm_comparison.png"))
    print("✅ Confusion Matrix 비교 시각화 저장 완료.")
except Exception as e:
    print(f"⚠️ Confusion Matrix 비교 실패: {e}")

print("✅ 분석 완료! 결과 파일들이 생성되었습니다.")
