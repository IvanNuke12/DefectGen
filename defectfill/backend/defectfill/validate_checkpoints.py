"""
validate_checkpoints.py
------------------------
Selects the best checkpoint from a DefectFill training run by measuring
generation quality against the real defect set.

Approach:
- For each checkpoint saved in <run_dir>/checkpoints/*.pt, generate a small
  FIXED batch (default 4 images) using the exact same generation pipeline as
  inference.py: same (good_image, mask) pairs from the generation plan and
  identical deterministic seeds (per-sample index), so the ONLY difference
  between checkpoints is the learned weights.
- Evaluate each generated batch with KID (quality / distribution distance to
  real defects) and IC-LPIPS (diversity across generated images).
- Select the best checkpoint: lowest KID among those whose IC-LPIPS is above a
  minimum threshold (avoids picking a collapsed generator that reproduces the
  same pattern with artificially low KID).

Writes:
  <output_dir>/status.json            (progreso para la web UI)
  <output_dir>/validation_results.json (tabla completa + mejor checkpoint)
"""

import os
import re
import json
import torch
import argparse
import numpy as np
from datetime import datetime
from types import SimpleNamespace

from inference import inference as run_inference
from evaluate import (
    KIDEvaluator,
    ICLPIPSEvaluator,
    collect_generated_images,
    collect_real_defect_images,
)


def write_status(output_dir, **kwargs):
    """Lightweight JSON status file used by the web UI to poll progress."""
    status_path = os.path.join(output_dir, "status.json")
    payload = {"updated_at": datetime.now().isoformat(), **kwargs}
    tmp_path = status_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, status_path)
    except Exception as e:
        print(f"[web-status] could not write status file: {e}")


def _ckpt_sort_key(path):
    """Ordena checkpoints por step numérico; checkpoint_final.pt va al final."""
    name = os.path.basename(path)
    m = re.search(r"checkpoint_(\d+)\.pt", name)
    if m:
        return int(m.group(1))
    return 10**12  # checkpoint_final.pt


def list_checkpoints(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    files.sort(key=_ckpt_sort_key)
    return [os.path.join(ckpt_dir, f) for f in files]


def _resolve_lora_from_checkpoint(ckpt_path, lora_rank, lora_alpha):
    """Usa los valores con los que se entrenó el checkpoint si están guardados."""
    from utils import read_checkpoint_config
    try:
        cfg = read_checkpoint_config(ckpt_path)
        if cfg:
            if cfg.get("lora_rank") is not None:
                lora_rank = cfg["lora_rank"]
            if cfg.get("lora_alpha") is not None:
                lora_alpha = cfg["lora_alpha"]
    except Exception as e:
        print(f"[validate] No se pudo leer config del checkpoint {os.path.basename(ckpt_path)}: {e}")
    return lora_rank, lora_alpha


def _real_defect_dir(data_dir, object_class, defect_type):
    train_dir = os.path.join(data_dir, object_class, "train", "defective", defect_type)
    if os.path.isdir(train_dir):
        return train_dir
    test_dir = os.path.join(data_dir, object_class, "test", "defective", defect_type)
    if os.path.isdir(test_dir):
        return test_dir
    return None


def generate_eval_batch(ckpt_path, args):
    """Genera un lote pequeño fijo con semillas deterministas idénticas entre
    checkpoints, reutilizando el pipeline completo de inference.py.

    Se guarda en <output_dir>/eval_<step>/ y se devuelve la lista de imágenes
    generadas (*_generated.png)."""
    step = _ckpt_sort_key(ckpt_path)
    eval_dir = os.path.join(args.output_dir, f"eval_{step}")
    os.makedirs(eval_dir, exist_ok=True)

    lora_rank, lora_alpha = _resolve_lora_from_checkpoint(
        ckpt_path, args.lora_rank, args.lora_alpha
    )

    inf_args = SimpleNamespace(
        checkpoint=ckpt_path,
        output_dir=eval_dir,
        object_class=args.object_class,
        class_name=args.class_name or args.object_class,
        defect_type=args.defect_type,
        data_dir=args.data_dir,
        image_path=None,
        image_dir=None,
        num_samples=args.num_samples,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        total_images=args.num_eval_images,
        batch_size=args.batch_size,
        use_compile=False,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        prompt=args.prompt,
        dilate_mask=args.dilate_mask,
        mask_kernel_size=args.mask_kernel_size,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_inference(inf_args)

    return collect_generated_images(eval_dir)


def main():
    parser = argparse.ArgumentParser(description="Validate DefectFill checkpoints (KID + IC-LPIPS)")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directorio del run de entrenamiento (contiene checkpoints/)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Raíz del dataset (PROJECTS_ROOT)")
    parser.add_argument("--object_class", type=str, required=True)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--defect_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directorio del job: status.json + validation_results.json")
    parser.add_argument("--num_eval_images", type=int, default=4,
                        help="Imágenes a generar por checkpoint (lote fijo, recomendado 2-5)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Candidatos por imagen (1 = determinista y barato)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--min_ic_lpips", type=float, default=0.05,
                        help="Umbral mínimo de diversidad; solo se consideran checkpoints por encima")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--dilate_mask", type=str, default="False")
    parser.add_argument("--mask_kernel_size", type=int, default=3)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.dilate_mask = args.dilate_mask.lower() == "true"
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = list_checkpoints(args.run_dir)
    if not checkpoints:
        write_status(args.output_dir, state="failed",
                     error="No hay checkpoints en " + os.path.join(args.run_dir, "checkpoints"))
        raise SystemExit(1)

    real_dir = _real_defect_dir(args.data_dir, args.object_class, args.defect_type)
    if not real_dir:
        write_status(args.output_dir, state="failed",
                     error=f"No se encuentra el conjunto real de defectos: "
                           f"{args.data_dir}/{args.object_class}/train/defective/{args.defect_type}")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[validate] device={device}, {len(checkpoints)} checkpoints, "
          f"eval_images={args.num_eval_images}, min_ic_lpips={args.min_ic_lpips}")
    print(f"[validate] real defect set: {real_dir}")

    kid = KIDEvaluator(device=device)
    iclpips = ICLPIPSEvaluator(device=device)

    results = []
    total = len(checkpoints)
    for i, ckpt in enumerate(checkpoints, start=1):
        step = _ckpt_sort_key(ckpt)
        label = f"checkpoint_{step}.pt" if step < 10**12 else "checkpoint_final.pt"
        print(f"\n===== [{i}/{total}] Validando {label} =====")
        write_status(args.output_dir, state="running",
                     step=i, total_steps=total, current=label)

        try:
            gen_images = generate_eval_batch(ckpt, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "checkpoint": ckpt, "step": step, "filename": label,
                "kid_mean": float("nan"), "kid_std": float("nan"),
                "ic_lpips_mean": float("nan"), "ic_lpips_std": float("nan"),
                "num_generated": 0, "num_real": 0, "error": str(e),
            })
            continue

        real_images = collect_real_defect_images(real_dir)
        print(f"  Generated: {len(gen_images)} | Real: {len(real_images)}")

        kid_mean = kid_std = ic_lpips_mean = ic_lpips_std = float("nan")
        if gen_images and real_images:
            kid_mean, kid_std = kid.compute_kid(
                real_images, gen_images,
                num_subsets=min(100, len(gen_images)),
                subset_size=min(len(gen_images), len(real_images)),
            )
            print(f"  KID: {kid_mean:.6f} ± {kid_std:.6f}")
        else:
            print("  Warning: no se pudo calcular KID (faltan imágenes generadas o reales)")

        if len(gen_images) >= 2:
            ic_lpips_mean, ic_lpips_std = iclpips.compute_ic_lpips(
                gen_images,
                max_pairs=min(1000, len(gen_images) * (len(gen_images) - 1) // 2),
            )
            print(f"  IC-LPIPS: {ic_lpips_mean:.6f} ± {ic_lpips_std:.6f}")
        else:
            print("  Warning: se necesitan >=2 generadas para IC-LPIPS")

        results.append({
            "checkpoint": ckpt,
            "step": step,
            "filename": label,
            "kid_mean": float(kid_mean),
            "kid_std": float(kid_std),
            "ic_lpips_mean": float(ic_lpips_mean),
            "ic_lpips_std": float(ic_lpips_std),
            "num_generated": len(gen_images),
            "num_real": len(real_images),
            "error": None,
        })

    # --- Selección: menor KID sujeto a IC-LPIPS mínimo ---
    valid = [r for r in results
             if not r["error"] and r["num_generated"] > 0
             and r["ic_lpips_mean"] == r["ic_lpips_mean"]  # not NaN
             and r["ic_lpips_mean"] >= args.min_ic_lpips]
    warning = None
    if not valid:
        valid = [r for r in results if not r["error"] and r["num_generated"] > 0]
        warning = (f"Ningún checkpoint superó IC-LPIPS >= {args.min_ic_lpips}; "
                   "se seleccionó el de menor KID sin aplicar el umbral de diversidad.")
    best = None
    if valid:
        best = min(valid, key=lambda r: r["kid_mean"])

    for r in results:
        r["best"] = bool(best and r["checkpoint"] == best["checkpoint"])

    payload = {
        "run_dir": args.run_dir,
        "object_class": args.object_class,
        "defect_type": args.defect_type,
        "num_eval_images": args.num_eval_images,
        "num_samples": args.num_samples,
        "min_ic_lpips": args.min_ic_lpips,
        "real_defect_dir": real_dir,
        "timestamp": datetime.now().isoformat(),
        "warning": warning,
        "results": results,
        "best": best["checkpoint"] if best else None,
    }
    out_path = os.path.join(args.output_dir, "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[validate] Resultados: {out_path}")

    if best:
        print(f"[validate] MEJOR CHECKPOINT: {best['filename']} "
              f"(KID {best['kid_mean']:.6f}, IC-LPIPS {best['ic_lpips_mean']:.6f})")
    if warning:
        print(f"[validate] AVISO: {warning}")

    write_status(args.output_dir, state="completed", step=total, total_steps=total,
                 object_class=args.object_class, defect_type=args.defect_type,
                 best=best["checkpoint"] if best else None,
                 warning=warning, results=results)


if __name__ == "__main__":
    main()
