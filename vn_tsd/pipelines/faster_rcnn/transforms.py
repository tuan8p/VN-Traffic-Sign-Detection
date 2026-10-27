"""Online-only tensor augmentations. Boxes use pixel-edge coordinates."""
from __future__ import annotations

from vn_tsd.data.preprocess_online import resolve_online_preprocess


def describe_effective(cfg):
    flags = resolve_online_preprocess(cfg)
    return dict(requested=cfg.get("online_preprocess", {}), resolved=flags,
        effective={k: v for k, v in flags.items() if k not in ("normalize", "letterbox")},
        already_satisfied={"letterbox": "processed640 cache", "normalize": "once inside model"})


def clip_target(target, size=640):
    result = dict(target)
    b = target["boxes"].clone().clamp_(0, size)
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    result["boxes"], result["labels"] = b[keep], target["labels"][keep]
    return result


def affine(image, target, scale, tx, ty):
    """Bilinear inverse warp; exact fractional translations, align_corners=False."""
    import torch
    import torch.nn.functional as F
    size = image.shape[-1]
    centers = torch.arange(size, dtype=image.dtype, device=image.device)+0.5
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    x = ((xx-size/2-tx)/scale+size/2)*2/size-1
    y = ((yy-size/2-ty)/scale+size/2)*2/size-1
    grid = torch.stack((x, y), -1).unsqueeze(0)
    fill = 114/255
    warped = F.grid_sample((image-fill).unsqueeze(0), grid, mode="bilinear",
                           padding_mode="zeros", align_corners=False).squeeze(0)+fill
    result = dict(target)
    b = target["boxes"].clone()
    b[:, [0, 2]] = scale*(b[:, [0, 2]]-size/2)+size/2+tx
    b[:, [1, 3]] = scale*(b[:, [1, 3]]-size/2)+size/2+ty
    result["boxes"] = b
    return warped.clamp(0, 1), clip_target(result, size)


class OnlineTransform:
    def __init__(self, cfg):
        self.flags = resolve_online_preprocess(cfg)
        self.params = cfg["online_preprocess"]
        if self.flags["hflip"]:
            raise ValueError("hflip prohibited")

    def prepare(self, dataset, index):
        import torch
        from torchvision.transforms import ColorJitter
        from torchvision.transforms.functional import resize, InterpolationMode
        image, target = dataset.raw(index)
        f, p = self.flags, self.params
        if not f["enabled"]:
            return image, target
        if f["mosaic"] and torch.rand(()).item() < p["mosaic_prob"]:
            tile = int(torch.randint(4, ()).item())
            indices = torch.randint(len(dataset), (4,)).tolist()
            indices[tile] = index
            canvas = torch.full_like(image, 114/255)
            boxes, labels = [], []
            for k, idx in enumerate(indices):
                im, t = dataset.raw(idx)
                x, y = (k % 2)*320, (k//2)*320
                canvas[:, y:y+320, x:x+320] = resize(im, [320, 320], InterpolationMode.BILINEAR, antialias=True)
                b = t["boxes"]/2
                b[:, [0, 2]] += x
                b[:, [1, 3]] += y
                boxes.append(b)
                labels.append(t["labels"])
            image = canvas
            target = dict(target, boxes=torch.cat(boxes), labels=torch.cat(labels))
        j = f["jitter"]
        if j:
            randoms = torch.rand(3).tolist()
            image, target = affine(image, target, 1-j+2*j*randoms[0],
                                    640*j*(2*randoms[1]-1), 640*j*(2*randoms[2]-1))
        if f["color_jitter"]:
            image = ColorJitter(**p["color_jitter_params"])(image)
        return image, clip_target(target)

    def __call__(self, dataset, index):
        import torch
        image, target = self.prepare(dataset, index)
        if self.flags["mixup"] and torch.rand(()).item() < self.params["mixup_prob"]:
            other, other_target = self.prepare(dataset, int(torch.randint(len(dataset), ()).item()))
            a = self.params["mixup_alpha"]
            weight = torch.distributions.Beta(a, a).sample().item()
            image = weight*image+(1-weight)*other
            target = dict(target, boxes=torch.cat((target["boxes"], other_target["boxes"])),
                          labels=torch.cat((target["labels"], other_target["labels"])))
        return image, target
