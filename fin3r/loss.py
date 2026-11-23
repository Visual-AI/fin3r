"""Self-view depth distillation and reference-camera pointmap regression."""

import torch
from torch import nn


def norm_depth(depth, mask):
    # Center and scale each view on the same valid teacher pixels.
    median = depth.masked_fill(~mask, float("nan")).flatten(-2).nanmedian(dim=-1).values
    centered = (depth - median.nan_to_num()[..., None, None]).masked_fill(~mask, 0)
    deviation = centered.abs().sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1)
    return centered / deviation.clamp_min(1e-6)[..., None, None], deviation


def confidence_loss(error, conf, mask, penalty):
    beta = conf.float().masked_fill(~mask, 1).clamp_min(1e-6)
    pixel = beta * error.masked_fill(~mask, 0) - penalty * beta.log()
    count = mask.sum((-1, -2))
    return pixel.sum((-1, -2)) / count.clamp_min(1)


class Fin3RLoss(nn.Module):
    def __init__(self, penalty=0.2):
        super().__init__()
        self.penalty = penalty

    def distillation(self, pred, batch):
        target = batch["teacher_depth"].float()
        mask = torch.isfinite(target) & (target > 0)
        teacher, spread = norm_depth(target, mask)
        student, _ = norm_depth(pred["depth"].squeeze(-1).float(), mask)
        usable = spread > 1e-6
        per_view = confidence_loss((student - teacher).square(), pred["depth_conf"], mask, self.penalty)
        return per_view.masked_fill(~usable, 0).sum() / usable.sum().clamp_min(1)

    def anchor_point_loss(self, pred, batch):
        target = batch["teacher_anchor_xyz"][:, 0].float()
        depth = batch["teacher_depth"][:, 0]
        mask = torch.isfinite(depth) & (depth > 0) & torch.isfinite(target).all(-1)
        count = mask.sum((-2, -1))
        def normalize(points):
            points = points.masked_fill(~mask[..., None], 0)
            radius = torch.linalg.vector_norm(points, dim=-1).sum((-2, -1)) / count.clamp_min(1)
            return points / radius.clamp_min(1e-6)[:, None, None, None]

        points = pred["world_points"][:, 0].float()
        error = torch.linalg.vector_norm(normalize(points) - normalize(target), dim=-1)
        loss = confidence_loss(error, pred["world_points_conf"][:, 0], mask, self.penalty)
        return loss.sum() / (count > 0).sum().clamp_min(1)

    def forward(self, pred, batch):
        zero = batch["images"].new_zeros(())
        distill = depth = point = anchor = zero
        if "teacher_depth" in batch:
            distill = self.distillation(pred, batch)
            if "teacher_anchor_xyz" in batch:
                anchor = self.anchor_point_loss(pred, batch)
        elif "gt_depth" in batch:
            # Geometry annotations have a common scale across all views.
            valid = batch["point_masks"]
            target = batch["gt_depth"]

            def relative_depth(d):
                z = d.masked_fill(~valid, 0)
                scale = z.sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)
                return z / scale.clamp_min(1e-6)[:, None, None, None]

            err = (relative_depth(pred["depth"][..., 0]) - relative_depth(target)).square()
            depth = confidence_loss(err, pred["depth_conf"], valid, self.penalty).mean()

            target_pts = batch["world_points"]
            pts = pred["world_points"].float()

            # One scale for the entire scene, including the non-reference views.
            def normalize(p):
                p = p.masked_fill(~valid[..., None], 0)
                length = torch.linalg.vector_norm(p, dim=-1)
                scale = length.sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)
                return p / scale.clamp_min(1e-6)[:, None, None, None, None]

            error = (normalize(pts) - normalize(target_pts)).square().sum(-1)
            point = confidence_loss(error, pred["world_points_conf"], valid, self.penalty).mean()
        else:
            raise ValueError("Missing supervision for this batch")
        return {
            "objective": distill + depth + point + anchor,
            "loss_anchor_xyz": anchor,
            "loss_distill": distill,
            "loss_gt_depth": depth,
            "loss_pointmap": point,
        }
