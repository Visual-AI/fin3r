import logging
from pathlib import Path

import torch
import torch.distributed as dist
from trainer import Trainer


def atomic_save(content, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(content, temp)
    temp.replace(path)


class Fin3RTrainer(Trainer):
    def _setup_env_variables(self, config):
        # The upstream helper logs the entire process environment.
        if config:
            import os

            os.environ.update(config)

    def _setup_components(self):
        super()._setup_components()
        if self.optim_conf.amp.amp_dtype == "bfloat16":
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)

    def _process_batch(self, batch):
        return batch  # GT point maps were already transformed into camera 0.

    def _step(self, batch, model, phase, loss_meters):
        anchor = "teacher_anchor_xyz" in batch
        pred = model(batch["images"], include_points=anchor or "world_points" in batch, anchor_points_only=anchor)
        losses = self.loss(pred, batch)
        b, s = batch["images"].shape[:2]
        unique = b if batch["source"] == "sa1b" else b * s
        weight = batch.get("loss_weight_count", unique)
        sizes = torch.tensor([unique, b, weight], device=self.device)
        dist.all_reduce(sizes)
        count, global_batch, total_weight = sizes.unbind()
        expected = getattr(self.train_dataset, "global_batch_size", None)
        if expected is not None:
            if self.accum_steps != 1 or int(global_batch) != expected:
                raise RuntimeError(f"Expected {expected} global sequences per update")
            if self.rank == 0 and self.steps[phase] % self.logging_conf.log_freq == 0:
                logging.info(
                    "update=%d sequences=%d source_images=%d",
                    self.steps[phase] + 1,
                    expected,
                    int(count),
                )
        weighted = losses["objective"] * (weight * dist.get_world_size() / total_weight)
        keys = ("loss_distill", "loss_gt_depth", "loss_pointmap", "loss_anchor_xyz")
        terms = torch.stack([losses[k].detach() for k in keys]) * weight
        dist.all_reduce(terms)
        terms /= total_weight
        if not torch.isfinite(terms).all():
            raise FloatingPointError(f"Non-finite global loss: {dict(zip(keys, terms.tolist()))}")
        # Report the same global loss on every rank while preserving the
        # per-rank derivatives used by DDP gradient averaging.
        losses["objective"] = terms.sum() + (weighted - weighted.detach())
        # Trainer logs the objective per optimizer update. Use the same meter weight here.
        for key, value in zip(keys, terms.tolist()):
            loss_meters[f"Loss/{phase}_{key}"].update(value, b)
        self.steps[phase] += 1
        return losses

    def save_checkpoint(self, epoch, checkpoint_names=None):
        updates = getattr(self.train_dataset, "updates_per_epoch", None)
        if updates is not None and self.steps["train"] != (epoch + 1) * updates:
            raise RuntimeError(f"Incorrect update count at checkpoint: {self.steps}")
        if self.rank == 0:
            folder = Path(self.checkpoint_conf.save_dir)
            state = self.model.module.adapter_state_dict()
            atomic_save(state, folder / f"lora_epoch_{epoch + 1:02}.pth")
            atomic_save(
                dict(
                    adapter=state,
                    epoch=epoch + 1,
                    steps=self.steps,
                    optimizer=self.optims[0].optimizer.state_dict(),
                    scaler=self.scaler.state_dict(),
                ),
                folder / "checkpoint.pt",
            )
            logging.info("Saved epoch %d", epoch + 1)
        dist.barrier()

    def _load_resuming_checkpoint(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_adapter(saved["adapter"])
        self.optims[0].optimizer.load_state_dict(saved["optimizer"])
        self.scaler.load_state_dict(saved["scaler"])
        self.epoch, self.steps = saved["epoch"], saved["steps"]
