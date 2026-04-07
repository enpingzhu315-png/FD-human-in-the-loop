#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:31:50 2026

@author: Enping Zhu

"""
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from models.fastvit import fastvit_t8
import math
from sklearn.svm import OneClassSVM
import torch.nn.functional as F
import random
import shutil


# 1. Dataset Definition
class TextImageDataset(Dataset):
    def __init__(self, image_folder, json_path, processor,r1,image_size=(256, 256)):
        self.image_folder = image_folder
        self.image_size = image_size
        with open(json_path, "r") as f:
            self.data = json.load(f)
        
        self.processor = processor
        
        self.r1 = r1
        
        self.word = ["Heat pipe","Electromagnetic pump","Cesium lake","Reactor core"]
        self.word = np.array(self.word)
        
        # Define image preprocessing
        self.preprocess1 = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.preprocess2 = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, str(item['image_ID'])+'.png')
        
        # Fault labels (heat pipe, electromagnetic pump, cesium generator, reactor core)
        a = torch.zeros([4])
        if "Heat pipe" in item["answers"][int(0)]:
            a[0] = 1
        if "Electromagnetic pump" in item["answers"][int(0)]:
            a[1] = 1
        if "Cesium lake" in item["answers"][int(0)]:
            a[2] = 1
        if "Reactor core" in item["answers"][int(0)]:
            a[3] = 1          
        
        # Load image and preprocess
        image = Image.open(image_path).convert("L")
        if self.r1 ==1:
            image = self.preprocess1(image)
        else:
            image = self.preprocess2(image)
        
        # Text encoding
        description = item["description"]
        tokens = self.processor(description,max_length=128,truncation=True,padding="max_length",return_tensors='pt')
        
        return image, tokens, a, item["answers"][int(0)]


# Dataset preprocessing - Data splitting
def Dataset_split(base_path,split_ratio=0.85):
    # Split images according to JSON, first read JSON file
    random.seed(42)
    
    path1 = os.path.join(base_path,"multi_data_0.05.json")
    
    with open(path1, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    random.shuffle(data)
    
    split_index = int(len(data) * split_ratio)
    
    part1 = data[:split_index]
    part2 = data[split_index:]
    
    json_file1 = os.path.join(base_path,"train_data.json")
    json_file2 = os.path.join(base_path,"text_data.json")
    
    with open(json_file1, 'w', encoding='utf-8') as f:
        json.dump(part1, f, ensure_ascii=False, indent=4)
    with open(json_file2, 'w', encoding='utf-8') as f:
        json.dump(part2, f, ensure_ascii=False, indent=4)
    
    # Split image folders
    image_base = os.path.join(base_path,"train_L/")
    image_file1 = os.path.join(base_path,"train_data/")
    image_file2 = os.path.join(base_path,"text_data/")
    
    # Delete if exists
    if os.path.exists(image_file1):
        shutil.rmtree(image_file1)
    os.makedirs(image_file1)
    if os.path.exists(image_file2):
        shutil.rmtree(image_file2)
    os.makedirs(image_file2)
    
    # Split
    for a in part1:
        dir1 = os.path.join(image_base, str(a["image_ID"])+".png")
        dir2 = os.path.join(image_file1, str(a["image_ID"])+".png")
        shutil.copy(dir1, dir2)
    
    for a in part2:
        dir1 = os.path.join(image_base, str(a["image_ID"])+".png")
        dir2 = os.path.join(image_file2, str(a["image_ID"])+".png")
        shutil.copy(dir1, dir2)
    
    return image_file1,json_file1,image_file2,json_file2
# Text tokenization
def word_processor():
    from tokenizers.models import BPE
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.normalizers import NFKC, Sequence, Lowercase
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    import copy
    from transformers import BertTokenizerFast

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = Sequence([
        Lowercase()
    ])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=2000, initial_alphabet=ByteLevel.alphabet(), special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>"
            ])
    tokenizer.train(["./processor/word.txt"], trainer)

    tokenizer.save(r"./processor/tokenizer.json")

    new_tokenizer = BertTokenizerFast.from_pretrained(r"./processor/")

    new_tokenizer.add_special_tokens({
      "eos_token": "</s>",
      "bos_token": "<s>",
      "unk_token": "<unk>",
      "pad_token": "<pad>",
      "mask_token": "<mask>"
    })
    
    return new_tokenizer

# Data splitting
def split_dataset(dataset,word1,r1=1):
    train = []
    val = []
    if isinstance(word1, (int, float, complex)):
        for j in range(r1):
            print(j,"/",r1)
            for i in range(len(dataset)):
                print(f"\r{i}/{len(dataset)}/", end='', flush=True)
                if np.random.rand()>word1:
                    val.append(dataset[i])
                else:
                    train.append(dataset[i])
        return train, val
    else:
        for i in range(len(dataset)):
            answers = dataset[i][3]
            if answers in word1:
                val.append(dataset[i])
            else:
                train.append(dataset[i])
        return train, val


# Statistical Analysis of Label Types in the Database
def label_sum(dataset):
    num = []
    for i in range(len(dataset)):
        num.append(dataset[i][2].numpy())
    num = np.array(num)
    num = np.unique(num,axis=0)
    return num


#%% 2. Network Design
## 2.1 Text Classification Network  (Human)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=490, d_model=64, nhead=4, num_layers=4, dim_feedforward=64, output_dim=128):
        super().__init__()
        
        # Embedding layer: Convert discrete IDs to vector representations
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer: Convert transformer output to desired output dimension
        self.output_layer = nn.Sequential(nn.Linear(d_model, d_model),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.5),
                                          nn.Linear(d_model, output_dim),
                                          )
        
        # Global pooling: Compress sequence into a single vector
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: [batch_size, 128]
        
        # Embedding + Positional Encoding
        x = self.embedding(x)  # [batch_size, 128, d_model]
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch_size, 128, d_model]
        
        # Global pooling: adjust dimensions for pooling
        x = x.transpose(1, 2)    # [batch_size, d_model, 128]
        x = self.global_pool(x)  # [batch_size, d_model, 1]
        x = x.squeeze(-1)        # [batch_size, d_model]
        
        # Project to output dimension
        x = self.output_layer(x)  # [batch_size, 32]
        return x


#%% 2.2 Cross-modal Multi-scale Feature Fusion Network
class MultimodalFusionClassifier(nn.Module):
    def __init__(self, model_image, model_text, num_classes=4):
        super(MultimodalFusionClassifier, self).__init__()
        
        # Freeze pretrained models
        self.model_image = model_image
        self.model_text = model_text
        for model in [self.model_image, self.model_text]:
            for param in model.parameters():
                param.requires_grad_(False)
        
        # Dimension alignment module
        self.image_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 64),  # Align to 64 dimensions
            nn.LayerNorm(64)
        )
        
        # Cross-attention (using 4-head attention, 64//4=16 per-head dimension)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 64*2=128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, texts):
        # Feature extraction (no gradients)
        with torch.no_grad():
            img_feats = self.model_image(images)  # [B,768]
            txt_feats = self.model_text(texts)    # [B,64]
        
        # Dimension alignment 768-->64
        img_feats = self.image_proj(img_feats)  # [B,64]
        
        # Cross-attention (image as query)
        attn_output, _ = self.cross_attn(
            query=img_feats.unsqueeze(1),  # [B,1,64]
            key=txt_feats.unsqueeze(1),
            value=txt_feats.unsqueeze(1))
        
        # Residual fusion
        fused_feats = torch.cat([
            img_feats,
            attn_output.squeeze(1)  # [B,64]
        ], dim=1)  # [B,128]
        
        return img_feats, txt_feats, fused_feats, self.classifier(fused_feats)


# Feature Layer Fusion Mechanism
# Feature-wise Linear Modulation
class FiLM(nn.Module):
    def __init__(self, in_txt=64, out_dim=768, hidden=128,
                 use_beta=False, gamma_scale=0.1, beta_scale=0.05):
        super().__init__()
        self.use_beta = use_beta
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.ln_txt = nn.LayerNorm(in_txt)

        self.g_net = nn.Sequential(
            nn.Linear(in_txt, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
        nn.init.zeros_(self.g_net[-1].weight); nn.init.zeros_(self.g_net[-1].bias)

        if use_beta:
            self.b_net = nn.Sequential(
                nn.Linear(in_txt, hidden), nn.GELU(),
                nn.Linear(hidden, out_dim)
            )
            nn.init.zeros_(self.b_net[-1].weight); nn.init.zeros_(self.b_net[-1].bias)

    def forward(self, x_img, x_txt):
        t = self.ln_txt(x_txt)
        gamma = 1.0 + self.gamma_scale * torch.tanh(self.g_net(t))
        if self.use_beta:
            beta = self.beta_scale * torch.tanh(self.b_net(t))
        else:
            beta = 0.0
        return gamma * x_img + beta


# ---- Simple Fusion Head: FiLM -> Pre-gating -> Classification (only the main output, including modality dropout)----
class SimpleFusionHead(nn.Module):
    """
    Lightweight Attention + Pre-gating fusion head.
    - FiLM -> proj -> modality dropout
    - cross-attention (zi queries zt) -> z_att
    - vector gate conditioned on [zi, zt, z_att] -> alpha [B,d]
    - fused: alpha * zi + (1-alpha) * z_att
    """
    def __init__(self, num_classes, in_img=768, in_txt=64, d=96,
                 gate_bias_init=-2.0, p_drop=0.5, film_use_beta=True,
                 p_img_drop=0.40, p_txt_drop=0.05, attn_heads=4, gate_temp=1.0,
                 return_attn=True):
        super().__init__()
        assert d % attn_heads == 0, "d must be divisible by attn_heads"

        self.ln_img = nn.LayerNorm(in_img)
        self.ln_txt = nn.LayerNorm(in_txt)
        self.film   = FiLM(in_txt=in_txt, out_dim=in_img, hidden=128,
                           use_beta=film_use_beta, gamma_scale=0.1, beta_scale=0.05)

        # projections
        self.img_proj = nn.Sequential(nn.Linear(in_img, d), nn.GELU(), nn.Dropout(0.2))
        self.txt_proj = nn.Sequential(nn.Linear(in_txt, d), nn.GELU(), nn.Dropout(0.2))

        # cross-attention (treat features as seq_len=1)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=attn_heads, batch_first=False)
        self.attn_refine = nn.Sequential(nn.Linear(d, d), nn.GELU())  # small refine

        # gate: lightweight MLP -> per-dim alpha
        self.gate = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        nn.init.constant_(self.gate[-1].bias, gate_bias_init)
        self.gate_temp = float(gate_temp)

        # classification head
        hid = max(64, d // 2)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hid), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hid, num_classes)
        )

        # modality dropout params
        self.p_img_drop = float(p_img_drop)
        self.p_txt_drop = float(p_txt_drop)

        # optionally return attention weights (for visualization)
        self.return_attn = return_attn

    def forward(self, feat_img, feat_txt, y=None):
        x_img = self.ln_img(feat_img)
        x_txt = self.ln_txt(feat_txt)

        # FiLM modulation (text -> image)
        x_img_mod = self.film(x_img, x_txt)

        # project to d
        zi = self.img_proj(x_img_mod)  # [B, d]
        zt = self.txt_proj(x_txt)      # [B, d]

        # modality dropout (training only)
        if self.training:
            B = zi.size(0); dev = zi.device
            keep_i = (torch.rand(B, 1, device=dev) > self.p_img_drop).float()
            keep_t = (torch.rand(B, 1, device=dev) > self.p_txt_drop).float()
            both_zero = (keep_i == 0) & (keep_t == 0)
            if both_zero.any():
                keep_t[both_zero] = 1.0
            if self.p_img_drop < 1.0:
                zi = zi * keep_i / (1.0 - self.p_img_drop)
            else:
                zi = torch.zeros_like(zi)
            if self.p_txt_drop < 1.0:
                zt = zt * keep_t / (1.0 - self.p_txt_drop)
            else:
                zt = torch.zeros_like(zt)

        # --- cross-attention: query=zi, key/value=zt ---
        # shape for nn.MultiheadAttention (S, B, E)
        q = zi.unsqueeze(0)  # [1, B, d]
        k = zt.unsqueeze(0)  # [1, B, d]
        v = zt.unsqueeze(0)  # [1, B, d]
        attn_out, attn_weights = self.attn(q, k, v, need_weights=True)  # attn_out: [1,B,d], attn_weights: [B, 1, 1]
        z_att = attn_out.squeeze(0)  # [B, d]
        # small residual refine
        z_att = self.attn_refine(z_att) + z_att

        # gate input: [zi, zt, z_att]
        gate_input = torch.cat([zi, zt, z_att], dim=-1)  # [B, 3*d]
        gate_logits = self.gate(gate_input) / max(1e-6, float(self.gate_temp))
        alpha = torch.sigmoid(gate_logits)  # [B, d] per-dim

        # Modify this part for sensitivity analysis if needed
        z = alpha * zi + (1.0 - alpha) * z_att  # [B, d]
        logits = self.cls_head(z)

        if self.return_attn:
            # attn_weights shape: (B, tgt_len=1, src_len=1) -> squeeze to (B,)
            aw = attn_weights.squeeze(-1).squeeze(-1)  # [B]
            return logits, aw, feat_img, feat_txt, z
        else:
            return logits, alpha


# ---- Model wrapping: Freezing feature extractors -> Simplified fusion head ----
class FullModel(nn.Module):
    """
    Assumptions:
      model_image(images)   -> [B, in_img]
      model_txt(text_input) -> [B, in_txt]
    Both models' parameters are frozen; only SimpleFusionHead is trained.
    """
    def __init__(self, model_image, model_txt, num_classes,
                 in_img=768, in_txt=64, d=96,
                 gate_bias_init=-2.5, film_use_beta=True,
                 p_img_drop=0.35, p_txt_drop=0.1,   # ★ New: pass dropout probability down
                 p_drop=0.5):
        super().__init__()
        self.img_enc = model_image.eval()
        self.txt_enc = model_txt.eval()
        for p in self.img_enc.parameters(): p.requires_grad = False
        for p in self.txt_enc.parameters(): p.requires_grad = False

        self.fusion = SimpleFusionHead(
            num_classes=num_classes, in_img=in_img, in_txt=in_txt, d=d,
            gate_bias_init=gate_bias_init, p_drop=p_drop, film_use_beta=film_use_beta,
            p_img_drop=p_img_drop, p_txt_drop=p_txt_drop
        )

    def forward(self, images, text_inputs):
        with torch.no_grad():
            feat_img = self.img_enc(images)        # [B, in_img]
            feat_txt = self.txt_enc(text_inputs)   # [B, in_txt]
        return self.fusion(feat_img, feat_txt)



#%% 2.3 Decision Layer Fusion for Fault Diagnosis
# ========== Random source dropout (Keep your implementation) ==========
class RowDropout(nn.Module):
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        batch_size, C, M, N = x.shape
        mask = (torch.rand((batch_size, 1, M, 1), device=x.device) > self.drop_prob).float()
        return x * mask

# ========== Backbone for feature extraction (Keep your implementation) ==========
class MiniBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout2d(0.1)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        return c1, c2, c3

class BiFPNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_c3 = nn.Conv2d(32, 16, 1)
        self.reduce_c2 = nn.Conv2d(32, 16, 1)
        self.reduce_c1 = nn.Conv2d(16, 16, 1)
        self.output_conv = nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, c1, c2, c3):
        p3 = self.reduce_c3(c3)
        p2 = self.reduce_c2(c2)
        p1 = self.reduce_c1(c1)
        p3_to_p2 = F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        p2 = p2 + p3_to_p2
        p2_to_p1 = F.interpolate(p2, size=p1.shape[-2:], mode='nearest')
        p1 = p1 + p2_to_p1
        p1_to_p2 = F.adaptive_max_pool2d(p1, output_size=p2.shape[-2:])
        p2 = p2 + p1_to_p2
        p2_to_p3 = F.adaptive_max_pool2d(p2, output_size=p3.shape[-2:])
        p3 = p3 + p2_to_p3
        out = F.interpolate(p3, size=c1.shape[-2:], mode='nearest')
        return self.output_conv(out)

# ========== Main Network: Add a ConflictGate on top (optional) ==========
class BiFPNNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.rowdrop = RowDropout(drop_prob=0.20)
        self.backbone = MiniBackbone()
        self.bifpn = BiFPNBlock()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(8, num_classes),
        )

    def forward(self, x_prob):
        """
        x_prob: [B,1,10,4] probability "image" (0=match, 1=machine, 2..9=humans)
        Returns: logits_final [B,4] (directly fed into BCEWithLogitsLoss)
        """
        # --- Save non-dropped input for gate use (RowDropout will modify x during training)
        x_prob_for_gate = x_prob.detach()  # Gate only does post-processing, no backprop to input
        # --- Pass through original FPN backbone
        x = self.rowdrop(x_prob)
        c1, c2, c3 = self.backbone(x)
        fused = self.bifpn(c1, c2, c3)
        return self.classifier(fused)


# ---------- Auxiliary: FPN Interpretable Post-processing ----------
from itertools import combinations
# ---------- Shape helper ----------
def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    Allows the following inputs:
      [1, h, 4] or [B, h, 4]  -> Automatically convert to [B, 1, h, 4]
      [B, 1, h, 4]            -> Return as is
    """
    if x.dim() == 3:
        # [B(or 1), h, 4]
        B, h, c = x.shape
        if c != 4:
            raise ValueError(f"Last dim must be 4, got {c}")
        return x.unsqueeze(1)  # -> [B,1,h,4]
    elif x.dim() == 4:
        # [B, 1, h, 4] or [B, h, 4, ?(illegal)]
        if x.shape[1] == 1 and x.shape[-1] == 4:
            return x
        else:
            raise ValueError(f"Expect [B,1,h,4], got {tuple(x.shape)}")
    else:
        raise ValueError(f"Expect x dim 3 or 4, got {x.dim()}")

@torch.no_grad()
def shap_iq_rows(
    model: nn.Module,
    x: torch.Tensor,                     # [1,h,4] / [B,h,4] / [B,1,h,4]
    y_true: torch.Tensor = None,         # [4] or [B,4]; can be None
    num_samples: int = 128,
    neutral_mode: str = "half",          # 'half'(0.5) | 'uniform'(0.25) | 'zero'(0.0)
    agg: str = "mean_pos",               # 'mean_pos' | 'sum_pos' | 'top1'
    machine_row: int = 1,
    human_rows: tuple = None             # Default None -> Automatically set to range(2, R)
):
    """
    Adaptive row-wise SHAP-IQ attribution for row number R=h (including second-order interaction and class-wise attribution).
    Returns fields consistent with the original, but no longer fixed at R=10.
    """
    # ---------- helpers ----------
    def _neutral_row(x_like: torch.Tensor, mode: str = "half") -> torch.Tensor:
        val = 0.5 if mode=="half" else (0.25 if mode=="uniform" else 0.0)
        return torch.full_like(x_like[:, :, 0:1, :], val)  # [B,1,1,4]

    @torch.no_grad()
    def _probs(model, x_in):  # -> [N,4]
        logits = model(x_in)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def _scalar_from_probs(p, y, top1_idx, how: str):
        # p: [N,4]; y: None or [N,4]; top1_idx: [N]
        if y is None or how == "top1":
            return p.gather(1, top1_idx.unsqueeze(1)).squeeze(1)
        pos_mask = (y > 0.5).to(p.dtype)
        pos_cnt  = pos_mask.sum(dim=-1)
        has_pos  = pos_cnt > 0
        f = torch.zeros(p.size(0), device=p.device, dtype=p.dtype)
        if has_pos.any():
            pos_sum = (p[has_pos] * pos_mask[has_pos]).sum(dim=-1)
            if how == "sum_pos":
                f[has_pos] = pos_sum
            else:
                f[has_pos] = pos_sum / pos_cnt[has_pos].clamp_min(1.0)
        if (~has_pos).any():
            f[~has_pos] = p[~has_pos].gather(
                1, top1_idx[~has_pos].unsqueeze(1)
            ).squeeze(1)
        return f


# ---------- Machine-Human Interaction Statistics (Adaptive R) ----------
@torch.no_grad()
def summarize_machine_vs_humans(
    I: torch.Tensor,                  # [B, R, R] from shap_iq_rows res["I"]
    machine_row: int = 1,            # Convention: 1=machine
    human_rows: tuple = None,        # None -> Automatically range(2, R)
    robust: bool = True,             # Whether to provide robust (trimmed) mean
    trim_frac: float = 0.125         # Trimming fraction
):
    """
    Returns a dict containing the following keys (each is a [B] tensor, except for 'label' which is a list):
      - mh_mean / mh_median / mh_sum / mh_strength / mh_pos_frac
      - mh_synergy_score ∈[-1,1]
      - mh_trimmed_mean (provided when robust=True)
      - label (either "cooperation" or "conflict")
      - mh_pairs: Original pair vector [B, |H|]
    """
    assert I.dim() == 3, f"Expect I shape [B,R,R], got {tuple(I.shape)}"
    B, R1, R2 = I.shape
    assert R1 == R2, f"I must be square [B,R,R], got {tuple(I.shape)}"
    device = I.device

    if human_rows is None:
        human_rows = tuple(range(2, R1))
    if machine_row < 0 or machine_row >= R1:
        raise ValueError(f"machine_row out of range: {machine_row} for R={R1}")
    for hr in human_rows:
        if hr < 0 or hr >= R1:
            raise ValueError(f"human_rows contains out-of-range index {hr} for R={R1}")

    if len(human_rows) == 0:
        # No human rows, return zero vector
        zeros = torch.zeros(B, device=device, dtype=I.dtype)
        return {
            "mh_pairs": torch.zeros(B, 0, device=device, dtype=I.dtype),
            "mh_mean": zeros, "mh_median": zeros, "mh_sum": zeros,
            "mh_strength": zeros, "mh_pos_frac": zeros,
            "mh_synergy_score": zeros, "mh_trimmed_mean": zeros,
            "label": ["cooperation"]*B
        }

    # Extract machine-human interaction values: I[:, machine, humans] -> [B, |H|]
    idx_list = list(human_rows)
    mh_pairs = I[:, machine_row][:, idx_list]  # [B, |H|]

    # Basic statistics
    mh_mean    = mh_pairs.mean(dim=1)
    mh_median  = mh_pairs.median(dim=1).values
    mh_sum     = mh_pairs.sum(dim=1)
    mh_strength= mh_pairs.abs().mean(dim=1)
    mh_pos_frac= (mh_pairs > 0).float().mean(dim=1)

    # Normalized cooperation-conflict score ∈[-1,1]
    pos_sum = mh_pairs.clamp_min(0).sum(dim=1)
    neg_sum = (-mh_pairs.clamp_max(0)).sum(dim=1)
    denom   = (mh_pairs.abs().sum(dim=1) + 1e-12)
    mh_synergy_score = (pos_sum - neg_sum) / denom

    # Robust (trimmed) mean: remove the extreme values from both ends, then compute the mean
    if robust and mh_pairs.size(1) >= 3:
        sorted_vals, _ = torch.sort(mh_pairs, dim=1)
        k = int(sorted_vals.size(1) * trim_frac)
        k = min(k, max(0, sorted_vals.size(1)//2 - 1))
        trimmed = sorted_vals[:, k:sorted_vals.size(1)-k] if k > 0 else sorted_vals
        mh_trimmed_mean = trimmed.mean(dim=1)
    else:
        mh_trimmed_mean = mh_mean

    labels = ["cooperation" if float(s) >= 0.0 else "conflict" for s in mh_synergy_score]

    return {
        "mh_pairs": mh_pairs,
        "mh_mean": mh_mean,
        "mh_median": mh_median,
        "mh_sum": mh_sum,
        "mh_strength": mh_strength,
        "mh_pos_frac": mh_pos_frac,
        "mh_synergy_score": mh_synergy_score,
        "mh_trimmed_mean": mh_trimmed_mean,
        "label": labels
    }

import copy
class ModelEMA:
    def __init__(self,model,decay=0.99):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self,model):
        msd = model.state_dict()
        for k,v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v*=self.decay
                v+=(1.-self.decay)*msd[k].detach()
    def forward(self,x):
        return self.ema(x)


#%% 3.0 Network Training and Validation
## 3.1.1 Text Network Training
def text_train(model,train_loader):
    # 1.1 Training configuration
    ii= 0 
    num_epochs=350
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    best_loss = 1000
    len1 = int(len(train_loader)*0.85)
    sigmod = nn.Sigmoid()
    pred_list = []
    # 1.2 Training starts
    for epoch in range(num_epochs):
        e2 = []
        step = 0
        for _, tokens, label, _ in train_loader:
            # Training part
            if step<len1:
                outputs = model(tokens["input_ids"].reshape([-1,128]).to(device))
                loss = criterion(outputs,label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                outputs = model(tokens["input_ids"].reshape([-1,128]).to(device))
                outputs = sigmod(outputs)
                outputs = (outputs > 0.70).float()
                loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
                e2.append(loss_train)
            step = step + 1
            
        if np.mean(e2) < best_loss:
            print(ii)
            ii = ii+1
            best_loss = np.mean(e2)
            torch.save(model.state_dict(),'model_text.pt')
            processor.save_pretrained(r'./processor')
        scheduler.step()
        pred_list.append(np.mean(e2))
        print("Study rate: ",optimizer.param_groups[0]["lr"]," ii ",ii,"loss_train:",loss.item(),"loss_valid:",np.mean(e2))
    
    model.load_state_dict(torch.load('model_text.pt',map_location=device))
    
    return model,pred_list


## 3.1.2 Text Network Validation
def text_val(model,val_loader,arr1):
    sigmod = nn.Sigmoid()
    e2 = []
    arr2 = np.zeros([len(arr1),2])
    for _, tokens, label, _ in val_loader:
        outputs = model(tokens["input_ids"].reshape([-1,128]).to(device))
        outputs = sigmod(outputs)
        outputs = (outputs > 0.70).float()
        loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
        e2.append(loss_train)
        i = np.where((arr1 == label.numpy()).all(axis=1))[0]
        arr2[i,0] = arr2[i,0]  + loss_train
        arr2[i,1] = arr2[i,1]  + 1
    e2 = np.array(e2)
    arr = np.concatenate((arr1, arr2), axis=1)
    return np.array(e2),arr

## 3.2.1 Image Network Training
def image_train(model,train_loader,val_loader,arr1):
    # 1.1 Training configuration
    ii= 0
    num_epochs=300
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.05)
    scheduler = StepLR(optimizer, step_size=150, gamma=0.5)
    best_loss = 1000
    len1 = int(len(train_loader)*0.85)
    sigmod = nn.Sigmoid()
    pred_list = []
    # 1.2 Training starts
    for epoch in range(num_epochs):
        e2 = []
        step = 0
        for image, _, label, _ in train_loader:
            # image = image[:,0,:,:].reshape([-1,1,256,256])
            if step<len1:
                model.train()
                outputs = model(image.to(device))
                loss = criterion(outputs,label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                model.eval()
                outputs = model(image.to(device))
                outputs = sigmod(outputs)
                outputs = (outputs > 0.70).float()
                loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
                e2.append(loss_train)
            step = step + 1
            
        if np.mean(e2) < best_loss:
            print(ii)
            ii = ii+1
            best_loss = np.mean(e2)
            torch.save(model.state_dict(),'model_image.pt')
        pred_list.append(np.mean(e2))
        print("Study rate: ",optimizer.param_groups[0]["lr"]," ii ",ii,"loss_train:",loss.item(),"loss_valid:",np.mean(e2))
        
        image_val(model,val_loader,arr1)
        
        if np.mean(e2) < 0.02:
            break

    return model,pred_list


## 3.2.2 Image Network Validation
def image_val(model,val_loader,arr1):
    sigmod = nn.Sigmoid()
    e2 = []
    arr2 = np.zeros([len(arr1),2])
    for image, _, label, _  in val_loader:
        outputs = model(image.to(device))
        outputs = sigmod(outputs)
        outputs = (outputs > 0.70).float()
        loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
        e2.append(loss_train)
        i = np.where((arr1 == label.numpy()).all(axis=1))[0]
        arr2[i,0] = arr2[i,0]  + loss_train
        arr2[i,1] = arr2[i,1]  + 1
    e2 = np.array(e2)
    arr = np.concatenate((arr1, arr2), axis=1)
    return np.array(e2),arr


## 3.3.1 Fusion Network Training
def fusion_train(model,train_loader):
    # 1.1 Training configuration
    ii= 0
    num_epochs=600
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=1e-6)
    best_loss = 1000
    len1 = int(len(train_loader)*0.80)
    sigmod = nn.Sigmoid()
    pred_list = []
    # 1.2 Training starts
    for epoch in range(num_epochs):
        e2 = []
        step = 0
        for image, tokens, label, _ in train_loader:
            if step<len1:
                model.train()
                image1= image.clone().to(device)
                noise1 = torch.normal(0, 0.025, size=image1.shape, device=image1.device)
                outputs,alpha,_,_,_ = model((image1+noise1).float(),tokens["input_ids"].reshape([-1,128]).to(device))
                loss = criterion(outputs,label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                model.eval()
                outputs,alpha,_,_,_ = model(image.to(device),tokens["input_ids"].reshape([-1,128]).to(device))
                outputs = sigmod(outputs)
                outputs = (outputs > 0.70).float()
                loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
                e2.append(loss_train)
            step = step + 1
        if np.mean(e2) < best_loss:
            print(ii)
            ii = ii+1
            best_loss = np.mean(e2)
            torch.save(model.state_dict(),'model_fusion.pt')
        # scheduler.step()
        pred_list.append(np.mean(e2))
        print("Study rate: ",optimizer.param_groups[0]["lr"]," ii ",ii,"loss_train:",loss.item(),"loss_valid:",np.mean(e2))
        
    model.load_state_dict(torch.load(r'model_fusion.pt',map_location=device))
    return model,pred_list


## 3.3.2 Fusion Network Validation
def fusion_val(model_fusion,val_loader,arr1):
    print("Fusion Network Validation")
    sigmod = nn.Sigmoid()
    e2 = []
    arr2 = np.zeros([len(arr1),2])
    arr3 = []
    len1 = len(val_loader)
    arr4 = np.zeros([len1,768])
    arr5 = np.zeros([len1,64])
    arr6 = np.zeros([len1,96])
    arr7 = np.zeros([len1,4])
    step = 0
    for image, tokens, label, _ in val_loader:
        outputs,alpha,f1,t1,z1 = model_fusion(image.to(device),tokens["input_ids"].reshape([-1,128]).to(device))
        outputs = sigmod(outputs)
        outputs = (outputs > 0.50).float()
        loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
        e2.append(loss_train)
        i = np.where((arr1 == label.numpy()).all(axis=1))[0]
        arr2[i,0] = arr2[i,0]  + loss_train
        arr2[i,1] = arr2[i,1]  + 1
        arr4[step,:] = f1.cpu().detach().numpy()
        arr5[step,:] = t1.cpu().detach().numpy()
        arr6[step,:] = z1.cpu().detach().numpy()
        arr7[step,:] = label.cpu().detach().numpy()
        step = step + 1
    e2 = np.array(e2)
    arr = np.concatenate((arr1, arr2), axis=1)
    return np.array(e2),arr,arr3,arr4,arr5,arr6,arr7


## 3.4.1 Multi-source Network Training
def BiFPNNet_train(model_fusion,model_image,model_oc,FPN,ema,train_loader,h):
    # 1.1 Training configuration
    ii= 0 
    num_epochs=300
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(FPN.parameters(), lr=8e-4, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=1e-5)
    best_loss = 1000
    len1 = int(len(train_loader)*0.80)
    sigmod = nn.Sigmoid()
    pred_list = []
    # 1.2 Training starts
    for epoch in range(num_epochs):
        e2 = []
        step = 0
        for _, _, label, _, inputs in train_loader:
            if step<len1:
                data = inputs.reshape([-1,1,h+2,4]).clone()
                noise = torch.normal(0, 0.1, size=data.shape, device=data.device)
                data = data + noise
                outputs = FPN(data.to(device))
                loss = criterion(outputs,label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.update(FPN)
            else:
                data = inputs.reshape([-1,1,h+2,4]).clone()
                noise = torch.normal(0, 0.1, size=data.shape, device=data.device)
                data = data + noise
                outputs = ema.ema(data.to(device))
                outputs = sigmod(outputs)
                outputs = (outputs > 0.70).float()
                loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
                e2.append(loss_train)
            step = step + 1
        if np.mean(e2) < best_loss:
            print(ii)
            ii = ii+1
            best_loss = np.mean(e2)
            torch.save(ema.ema.state_dict(),r'FPN_fusion.pt')
        scheduler.step()
        pred_list.append(np.mean(e2))
        print("Study rate: ",optimizer.param_groups[0]["lr"]," ii ",ii,"loss_train:",loss.item(),"loss_valid:",np.mean(e2))
    FPN.load_state_dict(torch.load('FPN_fusion.pt',map_location=device))
    return FPN,pred_list


## 3.4.2 Multi-source Network Validation
def multi_val(FPN,ema,val_loader,arr1,h,gate=False):
    sigmod = nn.Sigmoid()
    e2 = []
    arr2 = np.zeros([len(arr1),2])
    for _, _, label, _, inputs in val_loader:
        outputs = ema.ema(inputs.to(device).reshape([-1,1,h,4]))
        outputs = sigmod(outputs)
        outputs = (outputs > 0.70).float()
        loss_train = 1 - torch.all(outputs == label.to(device), dim=1).sum().item()/outputs.shape[0]
        e2.append(loss_train)
        if gate:
            if loss_train==1:
                print("Human:",inputs.reshape([10,4]))
                print("FPN:",outputs)
                print("True:",label)
        i = np.where((arr1 == label.numpy()).all(axis=1))[0]
        arr2[i,0] = arr2[i,0]  + loss_train
        arr2[i,1] = arr2[i,1]  + 1
    e2 = np.array(e2)
    print("Error:",sum(e2)/len(val_loader))
    arr = np.concatenate((arr1, arr2), axis=1)
    return np.array(e2),arr,sum(e2)/len(val_loader)


#%% 4.0 Fault Matching Network Training and Validation
# 4.1 Extract image features to create dataset
def get_data(train_set,val_set,model,juage=1):
    X_train = np.ones([len(train_set),768])
    X_val = np.ones([len(val_set),768])
    Y_train = np.ones([len(train_set)])
    if juage:
        Y_val = np.ones([len(val_set)])
    else:
        Y_val = np.ones([len(val_set)])*-1
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True,drop_last=True)
    step = 0
    for image, _, _, _ in train_loader:
        with torch.no_grad():
            outputs = model(image.to(device))
        outputs = outputs.cpu().detach().numpy()
        X_train[step,:] = outputs
        step = step + 1
    step = 0
    for image, _, _, _ in val_loader:
        with torch.no_grad():
            outputs = model(image.to(device))
        outputs = outputs.cpu().detach().numpy()
        X_val[step,:] = outputs
        step = step + 1
    return X_train,Y_train,X_val,Y_val


#%% 5. Multi-source Input Feature Judgment Network
# 5.1 Human Diagnosis Result Generation
def human_get(label_m,probability=0.7,human=5):
    row = torch.zeros([human,4])
    for i in range(human):
        if random.random() < probability:
            # Correct probability
            for j in range(4):
                if label_m[j] == 1:
                    if random.random() < 0.25:
                        row[i,j] = random.uniform(0.95, 1.0)
                    else:
                        row[i,j] = random.uniform(0.5, 0.85)
                else:
                    row[i,j] = random.uniform(0,0.5)
        else:
            # Error probability
            row[i,:] = torch.rand(4)
    return row

# 5.2 Data Generation for Multi-source Dataset
def multi_data(model_fusion,model_image,model_oc,train_set,val_set,probability=0.7,human=5,ph=0.1,couple_set = None):
    X_train = []
    X_val = []
    print("Generating multi-source training data")
    model_fusion.load_state_dict(torch.load(r'model_fusion.pt',map_location=device))
    for i in range(len(train_set)):
        
        image,tokens,label,fault = train_set[i]
        # 01 Machine Matching
        with torch.no_grad():
            outputs = model_image(image.reshape([-1,1,256,256]).to(device)).cpu().detach().numpy()
        data1 = ocsvm.predict(outputs)
        data1 = torch.ones([1,4])*max(data1[0],0)  # 1 for real, 0 for fake
        # 02 Machine Diagnosis
        with torch.no_grad():
            outputs,_,_,_,_ = model_fusion(image.reshape([-1,1,256,256]).to(device),tokens["input_ids"].reshape([-1,128]).to(device))
        data2 = sigmod(outputs).cpu()
        # 03 Human Diagnosis -------------------------------------------------- Set human fault probability and number of people
        data3 = human_get(label,probability,human)
        
        # 04 Concatenate data
        data = torch.cat([data1, data2, data3], dim=0)
        
        X_train.append((image,tokens,label,fault,data))
    print("Multi-source training data 1 generated")
    print("Generating multi-source validation data")
    for i in range(len(val_set)):
        
        image,tokens,label,fault = val_set[i]
        # 01 Machine Matching
        with torch.no_grad():
            outputs = model_image(image.reshape([-1,1,256,256]).to(device)).cpu().detach().numpy()
        data1 = ocsvm.predict(outputs)
        data1 = torch.ones([1,4])*max(data1[0],0)
        # 02 Machine Diagnosis
        with torch.no_grad():
            outputs,_,_,_,_ = model_fusion(image.reshape([-1,1,256,256]).to(device),tokens["input_ids"].reshape([-1,128]).to(device))
        data2 = sigmod(outputs).cpu()
        # 03 Human Diagnosis
        data3 = human_get(label,probability-ph,human)
        
        # 04 Concatenate data
        data = torch.cat([data1, data2, data3], dim=0)
        
        X_val.append((image,tokens,label,fault,data))
    print("Multi-source validation data generated")
    return X_train,X_val


#%%***********************************Main Program********************************************
if __name__ == '__main__':
    # 0. Data generation process
    # 1- Generalization Analysis 4- Feature Analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 0.2 Model Setup PNE
    processor = word_processor()
    # 0.3 Dataset creation
	'''
	The dataset is too large to upload in full, so only a small sample is provided here. 
	The complete dataset will be available on Hugging Face, 
	and the download link will be added later.
	'''
	
    val_set = torch.load("./val_set.pt")
    print("Validation Set Loading")
    
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,drop_last=True)
    arr1 = label_sum(val_set)
    

    # 1. Model definition
    sigmod = nn.Sigmoid()
    model_text = TransformerModel(vocab_size=processor.vocab_size,output_dim=4).to(device)
    model_image = fastvit_t8(pretrained=False)
    num_fr = model_image.head.in_features 
    model_image.head = nn.Sequential(
        nn.Linear(num_fr,num_fr),                  
        nn.ReLU(),
        nn.Linear(num_fr,4)                      
        )
    if True:
        model_image = model_image.to(device)
    else:
        model_image = torch.load("model_fastvit.pkl").to(device)
    
    # 2. Model Training
    # 2.1 Text Model Training
    if False:
        print("Text Feature Network Training")
        model_text,loss1 = text_train(model_text,train_loader)
        loss1 = np.array(loss1)
        np.savetxt("loss_text.txt", loss1,fmt="%.4f")
    else:
        print("Text Feature Network Import")
        model_text.load_state_dict(torch.load('model_text.pt',map_location=device))
        model_text.eval()
        num1,val1 = text_val(model_text,val_loader,arr1)
        np.savetxt("val_text.txt", val1,fmt="%.4f")
    # 2.2 Image Model Training
    if False:
        print("Image Feature Network Training")
        model_image,loss2 = image_train(model_image,train_loader,val_loader,arr1)
        loss2 = np.array(loss2)
        np.savetxt("loss_image.txt", loss2,fmt="%.4f")
        
        model_image.load_state_dict(torch.load('model_image.pt',map_location=device))
        model_image.eval()
        num2,val2 = image_val(model_image,val_loader,arr1)
        np.savetxt("val_image.txt", val2,fmt="%.4f")
    else:
        print("Image Feature Network Import")
        model_image.eval()
        model_image.load_state_dict(torch.load('model_image.pt',map_location=device))
        num2,val2 = image_val(model_image,val_loader,arr1)
        np.savetxt("val_image.txt", val2,fmt="%.4f")
    # 2.3 Cross-modal Multi-scale Fusion Classification
    # Keep only feature networks
    model_text.output_layer = nn.Identity()
    model_image.head = nn.Identity()
    
    model_fusion = FullModel(model_image, model_text, num_classes=4).to(device)

    if True:
        print("Fusion Feature Network Training")
        model_fusion,loss3 = fusion_train(model_fusion,train_loader)
        loss3 = np.array(loss3)
        np.savetxt("loss_fusion.txt", loss3,fmt="%.4f")
        model_fusion.eval()
        num3,val3,arr3,_,_,_,_ = fusion_val(model_fusion,val_loader,arr1)
        np.savetxt("val_fusion.txt", val3,fmt="%.4f")
    else:
        print("Fusion Network Import")
        model_fusion.load_state_dict(torch.load('model_fusion.pt',map_location=device))
        model_fusion.eval()
        num3,val3,arr3,arr4,arr5,arr6,arr7 = fusion_val(model_fusion,val_loader,arr1)
        np.savetxt("val_fusion.txt", val3,fmt="%.4f")
    
    ## 3. Failure Matching Model
    """
    Determine whether there are unknown faults in the dataset. 
    If there are no unknown faults, all values are 1, representing irrelevant information bits.
    """
    X_train,Y_train,X_val,Y_val = get_data(train_set,val_set,model_image)
    best_ga = 100
    for i in range(10):
        ga = 0.05+i*0.05
        ocsvm = OneClassSVM(kernel='rbf', gamma=ga, nu=0.015)
        ocsvm.fit(X_train)
        Y_pre  = ocsvm.predict(X_train)
        Y_pred = ocsvm.predict(X_val)
        num4 = 0
        for i in range(len(Y_val)):
            if Y_pred[i] == Y_val[i]:
                continue
            else:
                num4 = num4 + 1
        num5 = 0
        for i in range(len(Y_train)):
            if Y_pre[i] == Y_train[i]:
                continue
            else:
                num5 = num5 + 1
                
        e1 = 0.10*num5/len(Y_train)+0.90*num4/len(Y_val)
        if e1<best_ga:
            best_ga = num4/len(Y_val)
            e2 = ga
    
    
    ## 4. Multi-scale Feature Pyramid Fault Diagnosis
    # 4.1 Data Collection: Machine-Human-Matching
    '''
    Online generation of data based on the model, input image and human description, generate machine diagnostic results;
    Human fault diagnosis: impossible-small probability-possible-high probability-certain occurrence, fault diagnosis correct probability: 0.8
    Fault matching:
    '''
    del val_loader
    del train_loader
    
    
    p1 = 0.7
    h = 7
    if True:
        X_train,X_val = multi_data(model_fusion,model_image,ocsvm,train_set,val_set,probability=p1,ph=0,human=h)
        torch.save(X_train,"Decision_train.pt")
        torch.save(X_val,"Decision_val.pt")
        train_loader = DataLoader(X_train, batch_size=128, shuffle=True,drop_last=True)
        val_loader = DataLoader(X_val, batch_size=1, shuffle=False,drop_last=True)
    else:
        X_train = torch.load("Decision_train.pt")
        X_val = torch.load("Decision_val.pt")
        train_loader = DataLoader(X_train, batch_size=128, shuffle=True,drop_last=True)
        val_loader = DataLoader(X_val, batch_size=1, shuffle=False,drop_last=True)
    
    # 4.2 Decision Layer: Model Definition
    FPN = BiFPNNet().to(device)
    # 4.3 Decision Layer: Model Training
    if False:
        # 
        ema=ModelEMA(FPN,decay=0.70)
        FPN,loss4 = BiFPNNet_train(model_fusion,model_image,ocsvm,FPN,ema,train_loader,h)
        loss4 = np.array(loss4)
        np.savetxt("loss_multi.txt", loss4,fmt="%.4f")
        
        FPN.load_state_dict(torch.load('FPN_fusion.pt',map_location=device))
        FPN.eval()
        ema=ModelEMA(FPN,decay=0.70)
        ema.update(FPN)
        num4,val4,val5 = multi_val(FPN,ema,val_loader,arr1,h+2,gate=False) 
    else:
        # Interpretability analysis code
        FPN.load_state_dict(torch.load('FPN_fusion.pt',map_location=device))
        FPN.eval()
        ema=ModelEMA(FPN,decay=0.70)
        ema.update(FPN)
        num4,val4,val5 = multi_val(FPN,ema,val_loader,arr1,h+2,gate=False) 
        np.savetxt("val_multi.txt", val4,fmt="4%.f")
        p1 = random.randint(0, len(X_val))
        _, _, label, _, x = X_val[p1]
        res = shap_iq_rows(
            FPN,x.reshape([1,h+2,4]).to(device),
            y_true=None,
            num_samples=128,
            neutral_mode="half",
            agg="mean_pos"
            )
        phi = res["phi"][0]
        I = res["I"][0]
        
        m1 = np.array(res["machine_main_pc"][0].tolist())
        h1 = np.array(res["humans_main_pc"][0].tolist())
        p2 = m1 + h1
        
        print("Machine Contribution: ", m1/p2)
        print("Human Contribution: ", h1/p2)
        
        stats = summarize_machine_vs_humans(I.reshape([-1,I.shape[0],I.shape[1]]))
        print(stats["mh_mean"])
        
    print("Good job! It is free time!")
