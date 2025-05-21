import os
import gc
import time
import json
import torch
import argparse
import warnings
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import EmotionDataset, collate_fn_padded
from model import CrossGatedFusionClassifierV6_MultiGate
from train import train_model
from multiprocessing import freeze_support
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="D:/Datasets/Unified/train.csv")
    parser.add_argument("--val_path", type=str, default="D:/Datasets/Unified/val.csv")
    parser.add_argument("--lmdb_path", type=str, default="D:/Datasets/Unified/final_sequences.lmdb")
    parser.add_argument("--batch_size", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model_id", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    freeze_support()
    torch.backends.cudnn.benchmark = True

    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    MODEL_ID = args.model_id or f"V6PlusHybrid_{now_str}"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🧼 WandB 세션 정리
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project="emotion-v6", name=MODEL_ID)

    # 🧱 Dataset 로딩
    train_dataset = EmotionDataset(args.csv_path, args.lmdb_path)
    val_dataset = EmotionDataset(args.val_path, args.lmdb_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_padded,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_padded,
        num_workers=2,
        pin_memory=True
    )

    # 🧠 모델 초기화
    model = CrossGatedFusionClassifierV6_MultiGate(
        audio_dim=1024,
        text_dim=768,
        dims=[640, 512, 384, 320, 256],
        num_heads=8,
        num_classes=4,
        topk_audio=768,
        hidden_dim=256
    ).to(DEVICE)

    # ⏱️ 학습 시작
    start_time = time.time()
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        model_id=MODEL_ID,
        epochs=args.epochs,
        patience=7
    )

    # 📁 결과 저장 경로
    ckpt_dir = f"checkpoints/{MODEL_ID}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 📝 학습 로그 저장
    log_df = pd.DataFrame({
        "epoch": list(range(1, len(results["train_losses"]) + 1)),
        "train_loss": results["train_losses"],
        "val_f1": results["val_f1s"]
    })
    log_df.to_csv(os.path.join(ckpt_dir, "train_log.csv"), index=False, encoding="utf-8-sig")
    print(f"📝 학습 로그 저장 완료 → {ckpt_dir}/train_log.csv")

    # ⚙️ Config 저장
    config = {
        "model_id": MODEL_ID,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "topk_audio": 768,
        "dims": [640, 512, 384, 320, 256],
        "num_heads": 8,
        "train_csv": args.csv_path,
        "val_csv": args.val_path,
        "lmdb_path": args.lmdb_path,
        "device": str(DEVICE)
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 📑 args 별도 저장 (command-line 재현성)
    with open(os.path.join(ckpt_dir, "args.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # 💾 Alpha 저장
    fusion_alpha = torch.sigmoid(model.alpha_weights).detach().cpu().numpy().tolist()
    with open(os.path.join(ckpt_dir, "fusion_alpha.json"), "w", encoding="utf-8") as f:
        json.dump({"alpha": fusion_alpha}, f, indent=2)

    # ✅ 결과 출력
    best_epoch = 1 + int(torch.tensor(results["val_f1s"]).argmax())
    best_f1 = max(results["val_f1s"])
    print(f"\n🏆 Best Epoch: {best_epoch} | Val F1: {best_f1:.4f}")
    print(f"✅ Best model saved at: {results['best_model_path']}")

    elapsed = (time.time() - start_time) / 60
    print(f"⏱️ 총 학습 시간: {elapsed:.2f}분")

    # 🧼 정리
    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    main()
