import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VisionConfig:
    def __init__(
        self, in_channels, layers, out_channels, heads, n_embd, tokens, classes
    ):
        self.in_channels = in_channels
        self.layers = layers
        self.out_channels = out_channels
        self.heads = heads
        self.n_embd = n_embd
        self.tokens = tokens
        self.classes = classes


class AttentionPooling_2d(nn.Module):
    def __init__(
        self, tokens: int, n_embd: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embdding = nn.Parameter(
            torch.randn(tokens + 1, n_embd) / n_embd**0.5
        )
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, output_dim or n_embd)
        self.num_heads = num_heads

    def forward(self, x):
        x = rearrange(x, "b c h w -> (h w) b c")
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embdding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2d_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, stride=stride, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        self.acv1 = nn.ReLU()
        self.acv2 = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = self.acv1(x)
        x = self.conv2d_2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = identity + x
        return self.acv2(x)


class Block2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv2d_1 = Conv3x3(in_channels, out_channels, stride=stride)
        self.conv2d_2 = Conv3x3(out_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        return x


class ResNet2d_18(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.layers[0],
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.layer1 = Block2d(
            in_channels=config.layers[0],
            out_channels=config.layers[0],
            stride=2,
        )
        self.layer2 = Block2d(
            in_channels=config.layers[0],
            out_channels=config.layers[1],
            stride=2,
        )
        self.layer3 = Block2d(
            in_channels=config.layers[1],
            out_channels=config.layers[2],
            stride=2,
        )
        self.layer4 = Block2d(
            in_channels=config.layers[2],
            out_channels=config.layers[3],
            stride=2,
        )
        self.bn1 = nn.BatchNorm2d(config.layers[0])
        self.acv_fn = nn.ReLU()
        self.globalPooling = AttentionPooling_2d(
            tokens=config.tokens,
            n_embd=config.n_embd,
            num_heads=config.heads,
        )
        self.to_out = nn.Linear(self.config.out_channels, self.config.classes)

        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, AttentionPooling_2d):
            std = self.config.in_channels**-0.5
            nn.init.normal_(module.q_proj.weight, std=std)
            nn.init.normal_(module.k_proj.weight, std=std)
            nn.init.normal_(module.v_proj.weight, std=std)
            nn.init.normal_(module.c_proj.weight, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.02)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.acv_fn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.globalPooling(x)
        output = self.to_out(x)
        return output


def create_optimizers(model, config):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params())
    num_nodecay_params = sum(p.numel() for p in nodecay_params())

    print(
        f"num decayed params tensors: {len(decay_params)}, with {num_decay_params,} parameters"
    )
    print(
        f"num no_decay params tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
    )
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and config.device == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.init_lr, betas=config.betas, **extra_args
    )
    print(f"use fused AdamW:{use_fused}")
    return optimizer
