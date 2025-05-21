import os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter, deque
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import wandb
from visualize import save_alpha_and_gate_plot


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=-1)
        B = features.size(0)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        sim = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        sim = sim * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return -mean_log_prob_pos.mean()

# ✅ Gate Entropy Loss (per class gate)
def gate_entropy_loss(gate_weight):
    eps = 1e-8
    probs = torch.clamp(gate_weight, eps, 1.0 - eps)
    entropy = - (probs * torch.log(probs)).sum(dim=1)
    return entropy.mean()

# ✅ FocalLoss with per-class alpha and gamma
class FocalLoss(nn.Module):
    def __init__(self, num_classes, init_alpha, init_gamma, learnable_gamma=False):
        super().__init__()
        self.learnable_gamma = learnable_gamma
        self.num_classes = num_classes

        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha) + 1e-6))
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(init_gamma))) if learnable_gamma else torch.log(torch.tensor(init_gamma))

    def update_alpha(self, new_alpha):
        with torch.no_grad():
            self.log_alpha.data = torch.log(new_alpha + 1e-6).to(self.log_alpha.device)

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha = torch.exp(self.log_alpha).to(target.device)
        gamma = torch.exp(self.log_gamma).to(target.device) if self.learnable_gamma else self.log_gamma.to(target.device)
        alpha_t = alpha[target]
        gamma_t = gamma[target]
        loss = alpha_t * (1 - pt) ** gamma_t * ce_loss
        return loss.mean()

    def set_learnable_gamma(self):
        if not isinstance(self.log_gamma, nn.Parameter):
            self.log_gamma = nn.Parameter(self.log_gamma.clone().detach())
            self.learnable_gamma = True

class ClasswiseLossEMA:
    def __init__(self, num_classes, momentum=0.95):
        self.ema = torch.zeros(num_classes)
        self.momentum = momentum

    def update(self, losses, targets):
        for c in range(len(self.ema)):
            mask = (targets == c)
            if mask.any():
                class_loss = losses[mask].mean()
                self.ema[c] = self.momentum * self.ema[c] + (1 - self.momentum) * class_loss

    def get_alpha(self):
        return self.ema / (self.ema.sum() + 1e-6)

def contrastive_loss(a_avg, t_avg, temperature=0.1):
    a = F.normalize(a_avg, dim=-1)
    t = F.normalize(t_avg, dim=-1)
    logits = torch.matmul(a, t.T) / temperature
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def attention_entropy(attn_weights):
    # attn_weights: [B, C, T]
    eps = 1e-8
    ent = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # [B, C]
    return ent.mean()



    label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [C, C], 대각선 True
    positive = logits * label_mask.float()
    negative = logits * (~label_mask).float()

    # 대각선만 뽑아서 positive로 사용, 나머지는 negative
    pos_mean = positive.diagonal().mean()
    neg_mean = negative.sum() / (C * (C - 1))

    return -pos_mean + neg_mean

def compute_total_loss(outputs, targets, criterion, lambda_aux,
                       lambda_contrast=0.07, lambda_entropy=0.08,
                       lambda_cosin=0.07, lambda_query=0.01, lambda_query_ce=0.2, lambda_query_contrast=0.01):
    # 기본 손실들
    loss_main = criterion(outputs["logits"], targets)
    
    loss_cosine = criterion(outputs["cosine_logits"], targets)
    loss_pooled = criterion(outputs["pooled_logits"], targets)
    supcon = SupConLoss()
    feat = torch.cat([outputs["a_avg"], outputs["t_avg"]], dim=0)  # [2B, D]
    labels = targets.repeat(2)
    loss_contrast = supcon(feat, labels)
    loss_entropy = gate_entropy_loss(outputs["gate_weight"])

    # ✅ Query-specific 추가 손실
    
    loss_query_ent = attention_entropy(outputs["query_attn_weights"])  # [B, C, T]
    
    # ✅ Query Logit 대각선 loss (각 query→자기 emotion에 대한 confidence)
    diag_logits = torch.diagonal(outputs["query_logits"], dim1=1, dim2=2)  # [B, C]
    loss_query_diag = criterion(diag_logits, targets)

    return (
        loss_main
        + lambda_aux * (loss_cosine + loss_pooled)
        + lambda_contrast * loss_contrast
        + lambda_entropy * loss_entropy
        + lambda_query * loss_query_ent
        + lambda_query_ce * loss_query_diag
    )


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, scheduler, loss_tracker, device,
    lambda_aux, lambda_contrast=0.05, lambda_entropy=0.05, lambda_query=0.01,
    lambda_query_ce=0.2, lambda_query_contrast=0.01
):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for audio, text, audio_mask, text_mask, label in tqdm(loader, desc="📦 Train", ncols=100, leave=False):
        audio, text = audio.to(device), text.to(device)
        audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(audio, text, audio_mask, text_mask)

            # ✅ 게이트 가중치 통계 확인
            

            predicted = outputs["logits"].argmax(dim=1)
            raw_loss = F.cross_entropy(outputs["logits"], label, reduction='none')
            loss_tracker.update(raw_loss.detach(), label)
            loss = compute_total_loss(
                outputs, label, criterion, lambda_aux,
                lambda_contrast=lambda_contrast,
                lambda_entropy=lambda_entropy,
                lambda_query=0.01,
                lambda_query_ce=0.2,
                lambda_query_contrast=lambda_query_contrast
            )

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ⬇️ 이건 autocast 블록 바깥!
        total_loss += loss.item() * audio.size(0)
        correct += (predicted == label).sum().item()
        total += label.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(label.cpu().tolist())
        
       

    return total_loss / total, correct / total, all_preds, all_labels


def compute_f1_alpha(true, pred, num_classes=4):
    f1 = f1_score(true, pred, average=None, labels=list(range(num_classes)))
    diff = 1.0 - f1
    diff[diff < 0] = 0.0
    return diff / (diff.sum() + 1e-6)

def combine_alpha(loss_alpha, f1_alpha, weight=0.7):
    return (weight * loss_alpha + (1 - weight) * f1_alpha).detach()

def train_model(model, train_loader, val_loader, device, model_id="v6", epochs=30, patience=7):
    label_list = getattr(train_loader.dataset, "labels", None)
    class_counts = Counter(label_list)
    num_classes = len(class_counts)
    weights_raw = torch.tensor([1.0 / class_counts[i] for i in range(num_classes)], dtype=torch.float)
    weights = weights_raw / weights_raw.sum()
    init_gamma = [2.0, 2.0, 1.2, 1.2]
    criterion = FocalLoss(num_classes=4, init_alpha=weights.tolist(), init_gamma=init_gamma)

    loss_tracker = ClasswiseLossEMA(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 2 * len(train_loader), epochs * len(train_loader))
    scaler = GradScaler()

    wandb.init(project="emotion-v6", name=model_id)
    os.makedirs(f"checkpoints/{model_id}", exist_ok=True)
    os.makedirs(f"visuals/{model_id}", exist_ok=True)

    best_models = deque(maxlen=3)
    train_losses, val_f1s = [], []
    no_improve = 0

    for epoch in range(epochs):
        lambda_aux = 0.2

        avg_loss, train_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            loss_tracker, device, lambda_aux)

        f1_alpha = compute_f1_alpha(train_labels, train_preds, num_classes)
        loss_alpha = loss_tracker.get_alpha()
        hybrid_alpha = combine_alpha(loss_alpha, f1_alpha)

        if epoch < 10:
            hybrid_alpha[0] += 0.10  # sad
            hybrid_alpha[2] -= 0.05  # neutral
            hybrid_alpha[3] += 0.05  # happy
            hybrid_alpha = hybrid_alpha / (hybrid_alpha.sum() + 1e-6)
            criterion.update_alpha(hybrid_alpha.to(device))
        else:
            if not criterion.learnable_gamma:
                print(f"🔁 [Epoch {epoch+1}] Switching to learnable gamma.")
                criterion.set_learnable_gamma()

        train_losses.append(avg_loss)
        print(f"\n✅ Epoch {epoch+1}: Train Loss={avg_loss:.4f} | Acc={train_acc*100:.2f}% | α={hybrid_alpha.tolist()}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for audio, text, audio_mask, text_mask, label in tqdm(val_loader, desc="Val", ncols=100):
                audio, text = audio.to(device), text.to(device)
                audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)
                label = label.to(device)
                with autocast():
                    outputs = model(audio, text, audio_mask, text_mask)
                preds = outputs["logits"].argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(label.cpu().tolist())

        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        classwise = f1_score(val_labels, val_preds, average=None, labels=list(range(num_classes)))
        val_f1s.append(val_f1)
        print(f"✯ Validation F1-score: {val_f1*100:.2f}%\n📊 Classwise F1: {classwise.tolist()}")

        is_best = val_f1 >= best_models[0][0] - 1e-4 if best_models else True
        save_alpha_and_gate_plot(
            alpha_tensor=torch.exp(criterion.log_alpha),
            gate_tensor=outputs["gate_weight"],
            save_dir=f"visuals/{model_id}",
            epoch=epoch,
            log_alpha=criterion.log_alpha,
            is_best=is_best
        )

        ckpt_path = f"checkpoints/{model_id}/epoch{epoch+1}.pt"
        if len(best_models) < 3 or val_f1 > min(best_models)[0]:
            torch.save(model.state_dict(), ckpt_path)
            best_models.append((val_f1, ckpt_path))
            best_models = deque(sorted(best_models, reverse=True), maxlen=3)
            if val_f1 == best_models[0][0]:
                torch.save(model.state_dict(), f"checkpoints/{model_id}/best.ckpt")

        print(f"📂 Saved: {ckpt_path}")
        no_improve = 0 if is_best else no_improve + 1
        if no_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    return {
        "best_model_path": f"checkpoints/{model_id}/best.ckpt",
        "train_losses": train_losses,
        "val_f1s": val_f1s
    }