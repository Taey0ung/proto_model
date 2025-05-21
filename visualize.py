import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns

def save_alpha_and_gate_plot(alpha_tensor, gate_tensor, save_dir, epoch,
                             log_alpha=None, is_best=False,
                             attn_weights_a2t=None, attn_weights_t2a=None):
    if not is_best:
        return  # 베스트 에폭만 저장

    labels = ["sad", "angry", "neutral", "happy"]
    epoch_num = epoch + 1  # 1-based index

    def save_bar(values, title, ylabel, filename, color):
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=color)
        plt.ylim(0, 1.0)
        plt.title(f"[Epoch {epoch_num}] {title}")
        plt.xlabel("Emotion")
        plt.ylabel(ylabel)
        for i, val in enumerate(values):
            plt.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    def append_csv(path, row, header):
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    # ✅ 1. Fusion Alpha
    alpha_vals = alpha_tensor.detach().cpu().numpy()
    save_bar(alpha_vals, "Classwise Alpha (FocalLoss - normalized)", "Alpha Weight",
             f"alpha_epoch{epoch_num}.png", "skyblue")
    np.save(os.path.join(save_dir, f"fusion_alpha_epoch{epoch_num}.npy"), dict(zip(labels, alpha_vals)))
    append_csv(os.path.join(save_dir, "fusion_alpha.csv"), [epoch_num] + alpha_vals.tolist(), ["epoch"] + labels)

    # ✅ 2. Gate Mean
    if gate_tensor.ndim == 2 and gate_tensor.shape[1] == len(labels):
        gate_vals = gate_tensor.detach().cpu().numpy()
        gate_means = gate_vals.mean(axis=0)
        save_bar(gate_means, "Multi-Gate Mean by Class", "Gate Weight (mean)",
                 f"multi_gate_epoch{epoch_num}.png", "orchid")
        np.save(os.path.join(save_dir, f"gate_epoch{epoch_num}.npy"),
                {label: gate_vals[:, i] for i, label in enumerate(labels)})
        append_csv(os.path.join(save_dir, "gate_means.csv"), [epoch_num] + gate_means.tolist(), ["epoch"] + labels)

    # ✅ 3. Log(Alpha)
    if log_alpha is not None:
        log_vals = log_alpha.detach().cpu().numpy()
        save_bar(log_vals, "log(Alpha) Parameters", "log_alpha",
                 f"log_alpha_epoch{epoch_num}.png", "orange")

    # ✅ 4. Cross-modal Attention Heatmaps
    if attn_weights_a2t is not None and attn_weights_t2a is not None:
        for name, attn, cmap in zip(
            ["audio2text", "text2audio"],
            [attn_weights_a2t[0], attn_weights_t2a[0]],
            ["YlGnBu", "YlOrBr"]
        ):
            avg = attn.mean(0).detach().cpu().numpy()
            plt.figure(figsize=(6, 5))
            sns.heatmap(avg, cmap=cmap)
            plt.title(f"[Epoch {epoch_num}] Attention: {name.replace('2', ' → ')}")
            plt.xlabel(f"{name.split('2')[1].capitalize()} Tokens")
            plt.ylabel(f"{name.split('2')[0].capitalize()} Tokens")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"attn_{name}_epoch{epoch_num}.png"))
            plt.close()
