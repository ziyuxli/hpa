import os
import json
import argparse
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from eval_utils import layerwise_knn, layerwise_linear_probe, layerwise_fewshot_knn


def build_loader(data_root: str, model, batch_size: int, num_workers: int):
    cfg = resolve_data_config({}, model=model)
    tfm = create_transform(**cfg, is_training=False)

    ds = ImageFolder(root=data_root, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return ds, dl, cfg


def extract_layer_cls_embeddings(intermediates):
    cls_list = []
    for (patch_tokens, cls_tokens) in intermediates:
        assert cls_tokens.shape[1] == 1, cls_tokens.shape    
        assert patch_tokens.shape[1] == 196, patch_tokens.shape 
        cls = cls_tokens[:, 0, :]
        cls_list.append(cls) 
    
    return cls_list


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = timm.create_model(args.model, pretrained=True)
    model.eval().to(device)

    ds, dl, cfg = build_loader(args.data_root, model, args.batch, args.num_workers)

    class_id_to_name = {i: name for i, name in enumerate(ds.classes)}
    use_amp = bool(args.amp and device.type == "cuda")

    all_paths = []
    all_targets = []
    cls_by_layer = None 

    for batch_idx, (images, targets) in enumerate(dl):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            intermediates = model.forward_intermediates(
                images,
                indices=None,
                return_prefix_tokens=True,
                output_fmt="NLC",
                norm=True,
                intermediates_only=True,
            )

        cls_list = extract_layer_cls_embeddings(intermediates)
        cls_list = [c.detach().cpu() for c in cls_list]

        if cls_by_layer is None:
            cls_by_layer = [[] for _ in range(len(cls_list))]

        for l, cls in enumerate(cls_list):
            cls_by_layer[l].append(cls)

        start = batch_idx * dl.batch_size
        end = start + images.size(0)
        batch_paths = [ds.samples[i][0] for i in range(start, end)]

        all_paths.extend(batch_paths)
        all_targets.extend(targets.detach().cpu().tolist())

    cls_by_layer = [torch.cat(chunks, dim=0) for chunks in cls_by_layer]
    N, D = cls_by_layer[0].shape
    L = len(cls_by_layer)

    out_obj = {
        "model": args.model,
        "data_root": args.data_root,
        "paths": all_paths,
        "targets": torch.tensor(all_targets, dtype=torch.long),
        "class_id_to_name": class_id_to_name,
        "cls_by_layer": cls_by_layer,  # list length L, each [N, D]
        "preprocess_cfg": cfg,
        "note": "CLS embeddings per ViT block via timm.forward_intermediates(indices=None).",
        "shape": {"N": N, "L": L, "D": D},
    }
    return out_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_test/object_images_eval")
    ap.add_argument("--model", type=str, default="vit_base_patch16_224")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", type=str, default="things_vitb16_intermediates.pt")
    ap.add_argument("--eval_dir", type=str, default="eval_csv")
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_obj = run(args)

    print(
        "Embedding extracted.",
        f"Image Number={out_obj['shape']['N']},",
        f"Layer Number={out_obj['shape']['L']},",
        f"Dimension={out_obj['shape']['D']}",
    )

    targets = out_obj["targets"]
    cls_by_layer = out_obj["cls_by_layer"]

    # 统一 meta（每行都会带上这些信息）
    meta = {
        "model": args.model,
        "data_root": args.data_root,
        "N": out_obj["shape"]["N"],
        "L": out_obj["shape"]["L"],
        "D": out_obj["shape"]["D"],
    }


    eval_device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    # 1) layerwise knn
    layerwise_knn(
        cls_by_layer, targets,
        train_frac=args.train_frac, k=args.knn_k, seed=args.seed, device=eval_device,
        meta=meta,
        csv_path=os.path.join(args.eval_dir, "knn.csv"),
    )

    # 2) few-shot knn
    layerwise_fewshot_knn(
        cls_by_layer, targets,
        shots=args.shots, k=args.knn_k, episodes=args.episodes, seed=args.seed, device=eval_device,
        meta=meta,
        csv_path=os.path.join(args.eval_dir, "fewshot_knn.csv"),
    )


    layerwise_linear_probe(
        cls_by_layer, targets,
        train_frac=args.train_frac, seed=args.seed, device=eval_device,
        meta=meta,
        csv_path=os.path.join(args.eval_dir, "linear_probe.csv"),
    )


    # save embedding resultj
    torch.save(out_obj, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()