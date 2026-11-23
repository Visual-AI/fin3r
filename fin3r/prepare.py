import cv2
import numpy as np

WIDTH, HEIGHT = 518, 392


def crop(image, depth=None, intrinsic=None, canvas=None):
    h, w = image.shape[:2]
    tw, th = canvas or (WIDTH, HEIGHT)
    scale = min(tw / w, th / h) if canvas else max(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (tw - nw) // 2, (th - nh) // 2
    if not canvas:
        left, top = -((nw - tw) // 2), -((nh - th) // 2)

    def transform(arr, mode):
        arr = cv2.resize(arr, (nw, nh), interpolation=mode)
        if not canvas:
            return arr[-top : -top + th, -left : -left + tw]
        return cv2.copyMakeBorder(
            arr, top, th - nh - top, left, tw - nw - left, cv2.BORDER_CONSTANT, value=0
        )

    image = transform(image, cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if depth is not None:
        if depth.shape != (h, w):
            raise ValueError("Depth and RGB resolution differ")
        depth = transform(depth, cv2.INTER_NEAREST_EXACT)
    if intrinsic is not None:
        sx, sy = nw / w, nh / h
        affine = np.array([[sx, 0, (sx - 1) / 2 + left], [0, sy, (sy - 1) / 2 + top], [0, 0, 1]])
        intrinsic = affine @ intrinsic
    return image, depth, intrinsic


def tartanair_valid_depth(depth):
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1000)
    if valid.any():
        valid &= depth <= np.percentile(depth[valid], 98)
    return np.where(valid, depth, 0)
