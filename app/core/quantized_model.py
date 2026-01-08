"""
Инструментарий квантизации PyTorch-моделей.
1. AutoQuantizedModel – HF-совместимая обёртка (bitsandbytes, 4/8-bit).
2. Q4Linear – ручная 4-bit симметричная квантизация одного слоя.
3. save_q4_weights / load_q4_weights – сериализация собственных весов.
"""

from __future__ import annotations
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


# ------------------------------------------------------------------
# 1. HF-совместимая обёртка (bitsandbytes, 4-bit / 8-bit)
# ------------------------------------------------------------------
class AutoQuantizedModel:
    """
    Загружает любую HF-модель в 4-bit или 8-bit без ручного кода.
    Пример:
        model = AutoQuantizedModel.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            load_in_4bit=True
        )
    """

    @staticmethod
    def from_pretrained(
        model_id: str,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        **hf_kwargs
    ) -> AutoModelForCausalLM:
        if not BNB_AVAILABLE and (load_in_4bit or load_in_8bit):
            raise RuntimeError("bitsandbytes не установлен: pip install bitsandbytes")

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            hf_kwargs["quantization_config"] = qconfig
        elif load_in_8bit:
            hf_kwargs["load_in_8bit"] = True

        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **hf_kwargs
        )


# ------------------------------------------------------------------
# 2. Ручная 4-bit симметричная квантизация
# ------------------------------------------------------------------
@dataclass
class Q4Params:
    group_size: int = 128  # квантуем в группах по 128 весов
    qmin: int = -8         # 4-bit signed: [-8, 7]
    qmax: int = 7


class Q4Linear(nn.Module):
    """
    Квантизованный линейный слой (4-bit, симметричный, group-wise).
    Поддерживает `.to("cuda")` и `.to("cpu")` без переквантизации.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 qparams: Optional[Q4Params] = None):
        super().__init__()
        self.qparams = qparams or Q4Params()
        self.in_features = in_features
        self.out_features = out_features

        # 4-bit веса (packed: 2 значения в 1 байте)
        groups = (in_features * out_features + self.qparams.group_size - 1) // self.qparams.group_size
        self.register_buffer("qweight", torch.zeros((out_features * in_features + 1) // 2, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(groups, dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros(groups, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Разворачиваем packed 4-bit веса в float16
        weight_fp16 = dequantize_q4(self.qweight, self.scales, self.zeros,
                                    self.qparams.group_size, self.out_features, self.in_features)
        return torch.nn.functional.linear(x, weight_fp16, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, qparams: Optional[Q4Params] = None) -> "Q4Linear":
        """Квантуем обычный nn.Linear -> Q4Linear."""
        device = linear.weight.device
        qparams = qparams or Q4Params()
        out_f, in_f = linear.weight.shape
        q_layer = cls(in_f, out_f, linear.bias is not None, qparams).to(device)

        w = linear.weight.data.clone().to(torch.float32)
        groups = (in_f * out_f + qparams.group_size - 1) // qparams.group_size

        qweight_packed = []
        scales_list = []
        zeros_list = []

        for g in range(groups):
            start = g * qparams.group_size
            end = min(start + qparams.group_size, in_f * out_f)
            w_group = w.view(-1)[start:end]

            # symmetric scale
            scale = w_group.abs().max() / qparams.qmax
            scale = max(scale, 1e-6)
            q = torch.round(w_group / scale).clamp(qparams.qmin, qparams.qmax)
            zero = 0.0  # symmetric -> zero = 0

            # pack 4-bit
            q_int = q.to(torch.int8)
            packed = pack_i4(q_int)
            qweight_packed.append(packed)
            scales_list.append(scale.to(torch.float16))
            zeros_list.append(zero.to(torch.float16))

        q_layer.qweight = torch.cat(qweight_packed).to(device)
        q_layer.scales = torch.tensor(scales_list, dtype=torch.float16, device=device)
        q_layer.zeros = torch.tensor(zeros_list, dtype=torch.float16, device=device)
        if linear.bias is not None:
            q_layer.bias = linear.bias.data.to(torch.float16)
        return q_layer


# ------------------------------------------------------------------
# 3. Утилиты pack / unpack 4-bit
# ------------------------------------------------------------------
def pack_i4(tensor: torch.Tensor) -> torch.Tensor:
    """Сжимает int8 [-8..7] в packed uint8 (2 значения в 1 байте)."""
    assert tensor.dtype == torch.int8
    tensor = tensor & 0xF  # оставляем 4 бита
    if tensor.numel() % 2 == 1:
        tensor = torch.cat([tensor, torch.zeros(1, dtype=torch.int8, device=tensor.device)])
    low = tensor[0::2]
    high = tensor[1::2]
    packed = (low & 0xF) | ((high & 0xF) << 4)
    return packed.to(torch.uint8)


def unpack_i4(packed: torch.Tensor) -> torch.Tensor:
    """Распаковывает uint8 -> int8."""
    low = (packed & 0xF).to(torch.int8)
    high = ((packed >> 4) & 0xF).to(torch.int8)
    unpacked = torch.stack([low, high], dim=-1).view(-1)
    # переводим в signed: 0..15 -> -8..7
    return (unpacked & 0xF) - 8


def dequantize_q4(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                  group_size: int, out_f: int, in_f: int) -> torch.Tensor:
    unpacked = unpack_i4(qweight)  # int8
    scales = scales.to(torch.float32)
    zeros = zeros.to(torch.float32)
    groups = scales.numel()
    weight = torch.zeros(out_f * in_f, dtype=torch.float32, device=qweight.device)
    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, unpacked.numel())
        weight[start:end] = (unpacked[start:end].to(torch.float32) * scales[g]) + zeros[g]
    return weight.view(out_f, in_f).to(torch.float16)


# ------------------------------------------------------------------
# 4. Сохранение / загрузка квантизированных весов
# ------------------------------------------------------------------
@dataclass
class Q4StateDict:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: Optional[torch.Tensor]
    qparams: Q4Params


def save_q4_weights(module: Q4Linear, path: Path):
    """Сохраняет квантизированные веса в `.q4.pt`."""
    path.write_bytes(
        torch.save(
            {
                "qweight": module.qweight,
                "scales": module.scales,
                "zeros": module.zeros,
                "bias": module.bias,
                "qparams": module.qparams,
            }
        )
    )


def load_q4_weights(path: Path, device: str = "cpu") -> Q4Linear:
    """Загружает квантизированные веса и строит Q4Linear."""
    state = torch.load(path, map_location=device)
    qp = state["qparams"]
    # восстанавливаем shape из scales
    groups = state["scales"].numel()
    out_f = groups  # упрощённо: 1 группа = 1 выход
    in_f = qp.group_size
    layer = Q4Linear(in_f, out_f, bias=state["bias"] is not None, qparams=qp).to(device)
    layer.qweight = state["qweight"]
    layer.scales = state["scales"]
    layer.zeros = state["zeros"]
    layer.bias = state["bias"]
    return layer