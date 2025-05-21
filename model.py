# 🔹 Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 🔹 Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# 🔹 GRU-style Gating
class ResidualGRUGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.z = nn.Linear(dim * 2, dim)
        self.r = nn.Linear(dim * 2, dim)
        self.h_tilde = nn.Linear(dim * 2, dim)

    def forward(self, x, residual):
        concat = torch.cat([x, residual], dim=-1)
        z = torch.sigmoid(self.z(concat))
        r = torch.sigmoid(self.r(concat))
        h_tilde = torch.tanh(self.h_tilde(torch.cat([x, r * residual], dim=-1)))
        return (1 - z) * residual + z * h_tilde

# 🔹 Self Attention Block
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim * 2, dim), nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

# 🔹 Cross Attention Block
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = ResidualGRUGate(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim * 2, dim), nn.Dropout(0.1)
        )

    def forward(self, q, kv, mask_q=None, mask_kv=None):
        q_ = self.norm_q(q)
        kv_ = self.norm_kv(kv)
        attn_out, _ = self.attn(q_, kv_, kv_, key_padding_mask=mask_kv)
        gated = self.gate(attn_out, q)
        return gated + self.ffn(self.norm2(gated))

class CoAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm_a1 = nn.LayerNorm(dim)
        self.norm_t1 = nn.LayerNorm(dim)

        self.cross_attn_a2t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_t2a = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.gate_a = ResidualGRUGate(dim)
        self.gate_t = ResidualGRUGate(dim)

        self.norm_a2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)

        self.ffn_a = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.attn_weights_a2t = None
        self.attn_weights_t2a = None

    def forward(self, a, t, mask_a=None, mask_t=None):
    # Cross Attention + 저장
        qa = self.norm_a1(a)
        kt = self.norm_t1(t)
        vt = self.norm_t1(t)
        a2t, w_a2t = self.cross_attn_a2t(qa, kt, vt, key_padding_mask=mask_t)

        qt = self.norm_t1(t)
        ka = self.norm_a1(a)
        va = self.norm_a1(a)
        t2a, w_t2a = self.cross_attn_t2a(qt, ka, va, key_padding_mask=mask_a)


        self.attn_weights_a2t = w_a2t.detach()  # [B, num_heads, Q, K]
        self.attn_weights_t2a = w_t2a.detach()

    # Residual GRU Gating
        a = self.gate_a(a2t, a)
        t = self.gate_t(t2a, t)

    # Feed-forward + Norm + Residual
        a = a + self.ffn_a(self.norm_a2(a))
        t = t + self.ffn_t(self.norm_t2(t))

        return a, t
# 🔹 Cosine Classifier
class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.scale * torch.matmul(x, w.T)

class EmotionQueryPoolingHead(nn.Module):
    def __init__(self, dim, num_classes, use_mlp=False):
        super().__init__()
        self.num_classes = num_classes  # ✅ 이 줄 추가
        self.query_gen = nn.Sequential(
        nn.Linear(dim + num_classes, dim),
        nn.ReLU(),
        nn.Linear(dim, dim)
    )
        self.token_cls = nn.Linear(dim, num_classes)
        self.scale = dim ** -0.5

    def forward(self, x, mask=None, return_reg=False):
        B, T, D = x.shape
        C = self.num_classes  # or t.size(1) if input shape is [B, C, D]
        ctx_avg = x.mean(dim=1)  # [B, D]
        class_ids = F.one_hot(torch.arange(C, device=x.device), num_classes=C).float()  # [C, C]

        ctx_exp = ctx_avg.unsqueeze(1).expand(-1, C, -1)           # [B, C, D]
        class_ids_exp = class_ids.unsqueeze(0).expand(B, -1, -1)   # [B, C, C]

        query_input = torch.cat([ctx_exp, class_ids_exp], dim=-1)  # [B, C, D+C]
        query = self.query_gen(query_input)  # [B, C, D]

        attn_scores = torch.matmul(query, x.transpose(1, 2)) * self.scale  # [B, C, T]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, C, T]
        summary = torch.bmm(attn_weights, x)  # [B, C, D]

        query_logits = self.token_cls(summary)  # [B, C, C]
        logits = query_logits.diagonal(dim1=1, dim2=2)  # ✅ 개선: 클래스별 자기 자신 판단

        outputs = {
            "logits": logits,              # [B, C]
            "attn_weights": attn_weights,
            "query_logits": query_logits
        }

        if return_reg:
            return outputs

        return logits
# 🔹 CrossGatedFusionClassifierV6_MultiGate 
class CrossGatedFusionClassifierV6_MultiGate(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=768, dims=[640, 512, 384, 320, 256], num_heads=8, num_classes=4, topk_audio=1536, max_len=2048, hidden_dim=256):
        super().__init__()
        self.topk_audio = topk_audio
        self.num_classes = num_classes

        self.coattn = CoAttentionBlock(768, num_heads)
        self.cross_self_audio = SelfAttentionBlock(768, num_heads)
        self.cross_self_text = SelfAttentionBlock(768, num_heads)
        self.audio_proj = nn.Linear(audio_dim, 768)
        self.text_proj = nn.Linear(text_dim, 768)
        self.audio_conv = nn.Sequential(
            nn.Conv1d(768, 768, 3, padding=1), nn.ReLU(),
            nn.Conv1d(768, 768, 3, padding=1)
        )
        self.audio_pos = SinusoidalPositionalEncoding(768, max_len)
        self.text_pos = SinusoidalPositionalEncoding(768, max_len)

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(768, num_heads),
            CoAttentionBlock(768, num_heads),
            nn.Linear(768, dims[0])
        ])
        for i in range(len(dims)):
            self.blocks.append(SelfAttentionBlock(dims[i], num_heads) if i % 2 == 0 else CrossAttentionBlock(dims[i], num_heads))
            if i < len(dims) - 1:
                self.blocks.append(nn.Linear(dims[i], dims[i + 1]))
        
        fusion_input_dim = 4 * dims[-1]  # = 1024
        self.fusion_alpha_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.pooling_head = EmotionQueryPoolingHead(dims[-1], num_classes, use_mlp=True)
        self.cosine_cls = CosineClassifier(dims[-1], num_classes)
        
       # adaptive gate: 기존 대비 입력 크기 증가
        gate_input_dim = 3 * num_classes + 4 * dims[-1]
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        with torch.no_grad():
            self.gate[0].weight.zero_()
            self.gate[0].bias.zero_()

        self.final_gate = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )
        self.final_head = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.alpha_weights = nn.Parameter(torch.ones(num_classes))
        self.gamma_weights = nn.Parameter(torch.ones(num_classes))

    def select_topk_by_rms(self, seq, topk):
        B, L, D = seq.size()
        L1 = L // 2
        L2 = L - L1

        front = seq[:, :L1, :]
        back = seq[:, -L2:, :]

        rms_f = torch.norm(front, dim=-1)
        rms_b = torch.norm(back, dim=-1)

        kf = min(topk // 2, L1)
        kb = min(topk - kf, L2)

        idx_f = rms_f.topk(kf, dim=1).indices
        idx_b = rms_b.topk(kb, dim=1).indices + L1

        idx = torch.cat([idx_f, idx_b], dim=1)
        idx_sorted, _ = torch.sort(idx, dim=1)

        return torch.gather(seq, 1, idx_sorted.unsqueeze(-1).expand(-1, -1, D)), idx_sorted


    def forward(self, audio_seq, text_seq, audio_mask=None, text_mask=None):
        B, _, _ = audio_seq.shape
        B, Lt, _ = text_seq.shape
        audio_seq, topk_idx = self.select_topk_by_rms(audio_seq, self.topk_audio)
        La = audio_seq.size(1)
        audio_mask = torch.gather(audio_mask, 1, topk_idx) if audio_mask is not None else torch.zeros(B, La, dtype=torch.bool, device=audio_seq.device)
        text_mask = text_mask if text_mask is not None else torch.zeros(B, Lt, dtype=torch.bool, device=audio_seq.device)

        a = self.audio_proj(audio_seq)
        a = self.audio_conv(a.transpose(1, 2)).transpose(1, 2)
        a = self.audio_pos(a)
        t = self.text_proj(text_seq)
        t = self.text_pos(t)
        a, t = self.coattn(a, t, audio_mask, text_mask)
        a = self.cross_self_audio(a, audio_mask)
        t = self.cross_self_text(t, text_mask)
    
        for block in self.blocks:
            if isinstance(block, SelfAttentionBlock):
                a = block(a, audio_mask)
                t = block(t, text_mask)
            elif isinstance(block, CoAttentionBlock):
                a, t = block(a, t, audio_mask, text_mask)  # ✅ 수정: tuple unpack
            elif isinstance(block, nn.Linear):
                a = block(a)
                t = block(t)
        a_cls = a[:, 0, :]                      # 또는 torch.mean(a, dim=1)
        t_cls = t[:, 0, :]

        a_avg = a.masked_fill(audio_mask.unsqueeze(-1), 0).sum(1) / (~audio_mask).sum(1, keepdim=True)
        t_avg = t.masked_fill(text_mask.unsqueeze(-1), 0).sum(1) / (~text_mask).sum(1, keepdim=True)
        fusion_input = torch.cat([a_avg, t_avg, a_cls, t_cls], dim=-1)  # [B, 4D]
        alpha = self.fusion_alpha_mlp(fusion_input)       # [B, 1]
        fused = alpha * a_avg + (1 - alpha) * t_avg       # [B, D]

        x = torch.cat([a, t], dim=1)
        mask = torch.cat([audio_mask, text_mask], dim=1)
        pooling_outputs = self.pooling_head(x, mask, return_reg=True)
        pooled_logits = pooling_outputs["logits"]
        pooled_attn_weights = pooling_outputs["attn_weights"]
        query_logits = pooling_outputs["query_logits"]
        cosine_logits = self.cosine_cls(fused)
        query_diag_logits = torch.diagonal(query_logits, dim1=1, dim2=2)  # [B, C]
        gate_input = torch.cat([
            pooled_logits,        # [B, C]
            cosine_logits,        # [B, C]
            query_diag_logits,    # [B, C]
            a_avg,                # [B, D]
            t_avg,                # [B, D]
            a_cls,                # [B, D]
            t_cls                # [B, D]
        ], dim=-1)
        τ = 3.0
        gate_weight = torch.sigmoid(self.gate(gate_input) / τ)  # [B, C]  # ⬅️ 여기!  
        logits = gate_weight * cosine_logits + (1 - gate_weight) * pooled_logits
        # ⛳️ ✅ 바로 여기! → 이 logits에 파이널 게이트 달기
        
        gate_logits = self.final_gate(logits)  # [B, C]
        fusion_weights = F.softmax(gate_logits, dim=-1)
        logits = fusion_weights * logits + (1 - fusion_weights) * pooled_logits  # or cosine_logits
        final_logits = self.final_head(logits)

        return {
            "logits": final_logits,
            "cosine_logits": cosine_logits,
            "pooled_logits": pooled_logits,
            "a_avg": a_avg,
            "t_avg": t_avg,
            "gate_weight": gate_weight,
            "alpha_weights": self.alpha_weights,
            "gamma_weights": self.gamma_weights,
            "query_attn_weights": pooled_attn_weights,  # [B, C, T]
            "query_logits": query_logits  # [B, C, C]
        }
