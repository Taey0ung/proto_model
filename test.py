import torch
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from dataset import EmotionDataset, collate_fn_padded
from model import CrossGatedFusionClassifierV6_MultiGate
from tqdm import tqdm
from multiprocessing import freeze_support
from collections import Counter
import traceback

def main():
    MODEL_ID = "V6PlusHybrid_20250520_2157"
    TEST_CSV = "D:/Datasets/Unified/test.csv"
    LMDB_PATH = "D:/Datasets/Unified/final_sequences.lmdb"
    CKPT_PATH = f"checkpoints/{MODEL_ID}/best.ckpt"
    SAVE_DIR = f"results/{MODEL_ID}"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    label_names = ["sad", "angry", "neutral", "happy"]

    test_dataset = EmotionDataset(TEST_CSV, LMDB_PATH)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_padded,
        num_workers=2, pin_memory=True
    )

    model = CrossGatedFusionClassifierV6_MultiGate(
        audio_dim=1024,
        text_dim=768,
        dims=[640, 512, 384, 320, 256],
        num_heads=8,
        num_classes=4,
        topk_audio=768,
        hidden_dim=256
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_ids = [], [], []
    gate_weights_all = []

    with torch.no_grad():
        for audio, text, audio_mask, text_mask, label in tqdm(test_loader, desc="🚀 Inference"):
            audio, text = audio.to(DEVICE), text.to(DEVICE)
            audio_mask, text_mask = audio_mask.to(DEVICE), text_mask.to(DEVICE)
            label = label.to(DEVICE)

            outputs = model(audio, text, audio_mask, text_mask)
            preds = outputs["logits"].argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

            if "gate_weight" in outputs:
                gw = outputs["gate_weight"].detach().cpu()
                gate_weights_all.extend(gw.tolist())

            if hasattr(test_loader.dataset, "segment_ids"):
                all_ids.extend(test_loader.dataset.segment_ids[:len(preds)])

    print("[✅ Confusion Matrix 계산 중]")
    print("▶ 정답 분포 :", Counter(all_labels))
    print("▶ 예측 분포 :", Counter(all_preds))

    # Confusion matrix 시각화
    try:
        cm = confusion_matrix(all_labels, all_preds)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("[Test] Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(SAVE_DIR, "test_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[✅ Confusion Matrix 저장됨] → {cm_path}")
    except Exception as e:
        print("[❌ Confusion Matrix 저장 실패]")
        traceback.print_exc()

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)
    report_path = os.path.join(SAVE_DIR, "test_classification_report.csv")
    pd.DataFrame(report).transpose().to_csv(report_path, encoding="utf-8-sig")

    # F1 계산 및 summary 저장
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    classwise_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(len(label_names))))
    classwise_f1_dict = {label_names[i]: round(f, 4) for i, f in enumerate(classwise_f1)}

    with open(os.path.join(SAVE_DIR, "f1_scores.json"), "w", encoding="utf-8") as f:
        json.dump({
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "classwise_f1": classwise_f1_dict
        }, f, indent=2, ensure_ascii=False)

    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = os.path.getsize(CKPT_PATH) / (1024 ** 2)

    summary_path = os.path.join(SAVE_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "classwise_f1": classwise_f1_dict,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(model_size, 2)
        }, f, indent=2, ensure_ascii=False)

    print("[✅ 테스트 종료] 모든 결과 저장 완료")

if __name__ == "__main__":
    freeze_support()
    main()