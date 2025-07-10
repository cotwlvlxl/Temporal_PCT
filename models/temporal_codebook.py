import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class TemporalCodebook(nn.Module):
    """Vector quantization module for temporal residuals."""

    def __init__(self, token_dim, token_class_num, ema_decay=0.9):
        super().__init__()
        self.token_dim = token_dim
        self.token_class_num = token_class_num
        self.decay = ema_decay

        self.register_buffer('codebook', torch.empty(token_class_num, token_dim))
        self.codebook.data.normal_()
        self.register_buffer('ema_cluster_size', torch.zeros(token_class_num))
        self.register_buffer('ema_w', torch.empty(token_class_num, token_dim))
        self.ema_w.data.normal_()

    def forward(self, diff_feat, train=True):
        bs, dim = diff_feat.shape
        distances = (diff_feat.pow(2).sum(1, keepdim=True)
                     + self.codebook.pow(2).sum(1)
                     - 2 * torch.matmul(diff_feat, self.codebook.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(bs, self.token_class_num, device=diff_feat.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        quantized = torch.matmul(encodings, self.codebook)

        if train:
            dw = torch.matmul(encodings.t(), diff_feat.detach())
            n_encodings, n_dw = encodings.numel(), dw.numel()
            enc_shape, dw_shape = encodings.shape, dw.shape
            combined = torch.cat((encodings.flatten(), dw.flatten()))
            if dist.is_initialized():
                dist.all_reduce(combined)
            sync_enc, sync_dw = torch.split(combined, [n_encodings, n_dw])
            sync_enc, sync_dw = sync_enc.view(enc_shape), sync_dw.view(dw_shape)
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * sync_enc.sum(0)
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.token_class_num * 1e-5) * n
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
            self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)
            commit_loss = F.mse_loss(quantized.detach(), diff_feat)
            quantized = diff_feat + (quantized - diff_feat).detach()
        else:
            commit_loss = None
        return quantized, encoding_indices, commit_loss
