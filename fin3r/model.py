"""VGGT with the token-wise Q/K/V normalization used by Visual-AI/Fin3R."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vggt.models.vggt import VGGT


class RenormQKV(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        self.base = base
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        before = self.base(x)
        after = before + self.scale * self.B(self.A(x))
        original = before.unflatten(-1, (3, -1))
        combined = after.unflatten(-1, (3, -1))
        ratio = original.norm(dim=-1, keepdim=True) / (combined.norm(dim=-1, keepdim=True) + 1e-6)
        return (combined * ratio).flatten(-2)


class Fin3RVGGT(VGGT):
    def __init__(self, base_path, adapters=True, camera=False, head_chunk=4, olora=False):
        super().__init__(enable_camera=camera, enable_track=False)
        state = torch.load(base_path, weights_only=True, map_location="cpu", mmap=True)
        missing, extra = self.load_state_dict(state, strict=False)
        skipped = ("track_head.",) if camera else ("track_head.", "camera_head.")
        if missing or any(not key.startswith(skipped) for key in extra):
            raise ValueError(f"VGGT base mismatch: {missing}, {extra[:8]}")
        self.requires_grad_(False)
        self.head_chunk = head_chunk
        self.base_path = base_path
        self.olora = False
        if adapters:
            for block in self.aggregator.patch_embed.blocks:
                block.attn.qkv = RenormQKV(block.attn.qkv)
            self._set_olora(olora, state)
        elif olora:
            raise ValueError("OLoRA requires adapters")

    @torch.no_grad()
    def _set_olora(self, enabled, state=None):
        if enabled == self.olora:
            return
        # Always derive the frozen branch from the original FP32 CPU checkpoint.
        # This also handles loading different adapters on a reused model.
        if state is None:
            state = torch.load(self.base_path, weights_only=True, map_location="cpu", mmap=True)
        for i, block in enumerate(self.aggregator.patch_embed.blocks):
            layer = block.attn.qkv
            weight = state[f"aggregator.patch_embed.blocks.{i}.attn.qkv.weight"].float()
            if enabled:
                rank = layer.A.out_features
                q, r = torch.linalg.qr(weight, mode="reduced")
                b0, a0 = q[:, :rank].contiguous(), r[:rank].contiguous()
                layer.A.weight.copy_(a0)
                layer.B.weight.copy_(b0)
                weight = weight - (layer.scale * b0) @ a0
            layer.base.weight.copy_(weight)
        self.olora = enabled

    def adapter_weights(self):
        result = {}
        for i, block in enumerate(self.aggregator.patch_embed.blocks):
            for part in ("A", "B"):
                key = f"aggregator.patch_embed.blocks.{i}.attn.base_model.model.qkv.lora_{part}.default.weight"
                result[key] = getattr(block.attn.qkv, part).weight
        return result

    def adapter_state_dict(self):
        weights = {k: p.detach().cpu() for k, p in self.adapter_weights().items()}
        if self.olora:
            return {"format": "fin3r_olora_qr_v1", "weights": weights}
        return weights

    def load_adapter(self, state):
        olora = state.get("format") == "fin3r_olora_qr_v1"
        if olora:
            state = state["weights"]
        weights = self.adapter_weights()
        if weights.keys() != state.keys():
            raise ValueError("Expected the complete official Fin3R adapter state")
        for key, dest in weights.items():
            if dest.shape != state[key].shape:
                raise ValueError(key)
        self._set_olora(olora)
        with torch.no_grad():
            for key, dest in weights.items():
                dest.copy_(state[key])

    def _dense_head(self, head, tokens, images, start):
        # Frozen heads must still pass gradients to the encoder. Recompute only
        # one small group of images at a time, independently of sequence length.
        b, s = images.shape[:2]
        used = [tokens[i].flatten(0, 1) for i in head.intermediate_layer_idx]

        def run(rgb, *features):
            blocks = [None] * len(tokens)
            for idx, feat in zip(head.intermediate_layer_idx, features):
                blocks[idx] = feat[:, None].float()
            return head(blocks, rgb[:, None], start)

        results = []
        for begin in range(0, b * s, self.head_chunk):
            sl = slice(begin, begin + self.head_chunk)
            args = (images.flatten(0, 1)[sl], *(x[sl] for x in used))
            if self.training and torch.is_grad_enabled():
                result = checkpoint(run, *args, use_reentrant=False)
            else:
                result = run(*args)
            results.append(result)
        return tuple(torch.cat(group)[:, 0].unflatten(0, (b, s)) for group in zip(*results))

    def forward(self, images, *, include_depth=True, include_points=True, include_camera=True, anchor_points_only=False):
        tokens, start = self.aggregator(images)
        with torch.autocast(images.device.type, enabled=False):
            pred = {}
            if include_depth:
                pred["depth"], pred["depth_conf"] = self._dense_head(
                    self.depth_head, tokens, images, start
                )
            if include_points:
                point_tokens = [v[:, :1] if v is not None else None for v in tokens] if anchor_points_only else tokens
                point_images = images[:, :1] if anchor_points_only else images
                pred["world_points"], pred["world_points_conf"] = self._dense_head(
                    self.point_head, point_tokens, point_images, start
                )
            if include_camera and self.camera_head is not None:
                pred["pose_enc"] = self.camera_head([tokens[-1][:, :, :1].float()])[-1]
        return pred
