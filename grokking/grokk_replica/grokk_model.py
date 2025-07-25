from zlib import Z_BEST_SPEED
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer
from utils import causal_attn_mask, parameter_norm


class GrokkModel(nn.Module):
    def __init__(self, transformer_config, vocab_size, output_size, device):
        super(GrokkModel, self).__init__()
        self.transformer = Transformer(
            **transformer_config,
            vocab_size=vocab_size,
            output_size=output_size,
            hidden_layer=1
        )
        self.device = device

    def forward(self, x):
        attn_mask = (
            causal_attn_mask(x.shape[1])
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )
        predictions, attns, _, h = self.transformer(x, attn_mask)
        return predictions, attns

    def get_hidden(self, x):
        attn_mask = (
            causal_attn_mask(x.shape[1])
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )
        predictions, attns, _, hs = self.transformer(x, attn_mask, repeats=100)

        # a = torch.flatten(
        #     torch.flatten(torch.stack(attns, dim=1), start_dim=2), end_dim=1
        # )

        # h = torch.flatten(hidden, end_dim=1)

        a = torch.flatten(torch.stack(attns, dim=1), start_dim=1)
        # h = torch.flatten(hidden, start_dim=1)
        # z = torch.cat([a, h], dim=1)

        h = torch.cat(hs)
        h = torch.flatten(h, start_dim=1)

        return h, a

    def get_loss(self, x, y):
        predictions, attns = self(x)
        # print(torch.argmax(predictions[:, -1, :], dim=-1), x[:, -1])
        loss = F.cross_entropy(predictions[:, -1, :], y)
        accuracy = (torch.argmax(predictions[:, -1, :], dim=-1) == y).float().mean()
        attn_entropies = sum(
            [
                -(attn * torch.log(attn + 1e-7)).sum(dim=-1).mean().item()
                for attn in attns
            ]
        ) / len(attns)
        param_norm = parameter_norm(self)
        return loss, {
            "loss": (loss.item(), x.shape[0]),
            "accuracy": (accuracy.item(), x.shape[0]),
            "attn_entropy": (
                attn_entropies,
                len(attns) * x.shape[0] * (x.shape[1] - 1),
            ),
            "param_norm": (param_norm, 1),
        }
