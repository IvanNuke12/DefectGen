import os
import json
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from model import DefectFillModel
from utils import load_checkpoint, read_checkpoint_config, compute_spatial_lpips, compute_spatial_lpips_batch
from torchvision.utils import save_image
from torchvision import transforms


def write_status(output_dir, **kwargs):
    """Lightweight JSON status file used by the web UI to poll inference progress."""
    status_path = os.path.join(output_dir, "status.json")
    payload = {"updated_at": datetime.now().isoformat(), **kwargs}
    tmp_path = status_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, status_path)
    except Exception as e:
        print(f"[web-status] could not write status file: {e}")


def smart_crop_dynamic(image, mask, base_size=512):
    """
    Crops the image to fit the defect (classic SQUARE crop, consistent with the
    model it was trained with).
    - If the square crop <= base_size: resized to base_size (INTER_AREA/INTER_CUBIC).
    - If the square crop > base_size (large/panoramic defect): returned at
      NATURAL resolution so the downstream TILING splits it into base tiles
      (instead of distorting a big square into base_size).
    Returns (crop_img, crop_mask, (orig_h, orig_w)) where orig are the NATURAL
    crop dimensions, and crop_img/crop_mask are either natural (only when >base)
    or already resized to base_size x base_size.
    """
    h, w = image.shape[:2]

    # Find the Bounding Box of the defect
    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) == 0:
        # No defect? Center square of base_size
        cy, cx = h // 2, w // 2
        crop_size = base_size
    else:
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)

        defect_h = max_y - min_y
        defect_w = max_x - min_x

        cy = min_y + defect_h // 2
        cx = min_x + defect_w // 2

        # Box big enough to hold the defect + context padding; at least base_size
        max_dim = max(defect_h, defect_w)
        padding = 50
        crop_size = max(base_size, max_dim + padding)

    # Square crop coordinates centered on the defect
    half_size = crop_size // 2
    x1 = cx - half_size
    y1 = cy - half_size
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Shift box back into image bounds
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    orig_h, orig_w = crop_img.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("Crop vacío: comprueba la máscara y las dimensiones de la imagen.")

    # If the square crop exceeds base_size -> keep at NATURAL resolution for
    # tiling (avoids distorting a large square into base_size).
    if orig_h > base_size or orig_w > base_size:
        return crop_img, crop_mask, (orig_h, orig_w)

    # Otherwise resize to base_size (classic behavior, consistent with training)
    if orig_h != base_size or orig_w != base_size:
        if orig_h > base_size or orig_w > base_size:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        crop_img = cv2.resize(crop_img, (base_size, base_size), interpolation=interp)
        crop_mask = cv2.resize(crop_mask, (base_size, base_size), interpolation=cv2.INTER_NEAREST)

    return crop_img, crop_mask, (orig_h, orig_w)


def count_available_resources(data_dir, object_class, defect_type):
    """Counts available good images and reference masks for synthetic generation."""
    # Good images directory
    good_dir = os.path.join(data_dir, object_class, "test", "good")
    num_good_images = len([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(good_dir) else 0
    
    # Mask directory (prioritize training masks for reference)
    train_mask_dir = os.path.join(data_dir, object_class, "train", "defective_masks", defect_type)
    test_mask_dir = os.path.join(data_dir, object_class, "test", "defective_masks", defect_type)
    
    if os.path.exists(train_mask_dir):
        num_masks = len([f for f in os.listdir(train_mask_dir) if f.endswith('.png')])
        mask_dir = train_mask_dir
    elif os.path.exists(test_mask_dir):
        num_masks = len([f for f in os.listdir(test_mask_dir) if f.endswith('.png')])
        mask_dir = test_mask_dir
    else:
        num_masks = 0
        mask_dir = None
    
    return num_good_images, num_masks, good_dir, mask_dir


def calculate_generation_plan(num_good_images, num_masks, target_total=100):
    """Calculates a combination plan of good images and masks to reach the target total."""
    if num_good_images == 0 or num_masks == 0:
        return []
    
    generation_plan = []
    output_idx = 0
    
    # Loop through good images and masks until target count is met
    while output_idx < target_total:
        for mask_idx in range(num_masks):
            if output_idx >= target_total:
                break
            good_idx = output_idx % num_good_images  # Cycle through good images
            generation_plan.append((good_idx, mask_idx, output_idx))
            output_idx += 1
    
    return generation_plan


def inference(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== Unified FP16 Precision Configuration ==========
    dtype = torch.float16
    
    # Enable TF32 acceleration (Ampere+ architectures: RTX 30/40/50 series)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: {device}, dtype: {dtype}, TF32: enabled")
    
    def resolve_prompt(custom_prompt, object_class):
        """Builds the generation prompt: custom if provided, else the training-consistent default.
        The learned <defect> token is mandatory; it is auto-appended when missing."""
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
            if model.placeholder_token not in prompt:
                print(f"[Info] Prompt '{prompt}' lacks the learned token {model.placeholder_token}. Appending it.")
                prompt = f"{prompt} with {model.placeholder_token}"
            return prompt
        return f"A {args.class_name} with {model.placeholder_token}"
    
    # Match LoRA config with the training checkpoint: the model MUST be built with
    # the same lora_rank/lora_alpha used during training or loading will fail.
    if args.checkpoint:
        ckpt_cfg = read_checkpoint_config(args.checkpoint)
        if ckpt_cfg and ckpt_cfg.get("lora_rank") is not None:
            if ckpt_cfg["lora_rank"] != args.lora_rank:
                print(f"[Info] Checkpoint trained with lora_rank={ckpt_cfg['lora_rank']} "
                      f"(requested --lora_rank={args.lora_rank}). Using checkpoint value.")
                args.lora_rank = ckpt_cfg["lora_rank"]
        if ckpt_cfg and ckpt_cfg.get("lora_alpha") is not None:
            if ckpt_cfg["lora_alpha"] != args.lora_alpha:
                print(f"[Info] Checkpoint trained with lora_alpha={ckpt_cfg['lora_alpha']} "
                      f"(requested --lora_alpha={args.lora_alpha}). Using checkpoint value.")
                args.lora_alpha = ckpt_cfg["lora_alpha"]
        print(f"[Info] Using lora_rank={args.lora_rank}, lora_alpha={args.lora_alpha}")

    # Initialize model
    model = DefectFillModel(
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Ensure VAE is also in FP16 to save VRAM
    model.pipeline.vae.to(dtype=dtype)
    
    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, None, args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Set to evaluation mode
    model.pipeline.unet.eval()
    model.pipeline.text_encoder.eval()
    


    def fixed_inference_batch(model, clean_image, mask, object_class, defect_type,
                              num_samples=8, steps=50, guidance_scale=7.5,
                              batch_size=4, custom_prompt=None, seed_offset=0):
        """
        Performs inference using the custom model.generate() method.
        Ensures consistency between training and inference phases.
        """
        prompt = resolve_prompt(custom_prompt, object_class)
        
        print(f"Using prompt: '{prompt}'")
        
        _, _, h_input, w_input = clean_image.shape
        BASE = 512

        # Sin tiling: cualquier crop se ajusta a 512x512, como en el pipeline
        # de test del entrenamiento. Evita los tiles no cuadrados y los
        # tamaños no múltiplos de 8 del VAE.
        if (h_input, w_input) != (BASE, BASE):
            print(f"Crop {w_input}x{h_input} -> resize a {BASE}x{BASE} (sin tiling)")
            clean_image = torch.nn.functional.interpolate(
                clean_image, size=(BASE, BASE), mode='bilinear', align_corners=False)
            mask = torch.nn.functional.interpolate(
                mask, size=(BASE, BASE), mode='nearest')
            h_input = w_input = BASE
        # La máscara puede llegar con otro tamaño que la imagen (p. ej.
        # máscara de referencia sin alinear): forzarla a BASE igualmente.
        mh, mw = mask.shape[-2:]
        if (mh, mw) != (BASE, BASE):
            print(f"Máscara {mw}x{mh} -> resize a {BASE}x{BASE}")
            mask = torch.nn.functional.interpolate(
                mask, size=(BASE, BASE), mode='nearest')

        print(f"Generating {num_samples} samples (batch_size={batch_size}, steps={steps})")
        
        # ========== Phase 1: Batch Sample Generation ==========
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            print(f"Batch {batch_idx+1}/{num_batches}: Generating samples {start_idx+1}-{end_idx}")
            
            batch_clean = clean_image.repeat(current_batch_size, 1, 1, 1)
            batch_mask = mask.repeat(current_batch_size, 1, 1, 1)
            
            # Use deterministic seed per sample for reproducibility (offset allows multiple outputs per mask)
            generator = torch.Generator(device=device).manual_seed(start_idx + seed_offset)
            
            # Consistent with training: 9-channel input + CFG + iterative bg preservation
            batch_samples = model.generate(
                image=batch_clean,
                mask=batch_mask,
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            # Convert back to [-1, 1] range for LPIPS (generate returns [0, 1])
            batch_samples_model_format = (batch_samples * 2.0) - 1.0
            all_samples.append(batch_samples_model_format)
        
        samples_model_format = torch.cat(all_samples, dim=0)
        
        if samples_model_format.shape[-2:] != (h_input, w_input):
            samples_model_format = torch.nn.functional.interpolate(
                samples_model_format, size=(h_input, w_input), mode='bilinear'
            )
        
        # ========== Phase 2: Batch LPIPS Selection ==========
        mask_resized = mask if mask.shape[-2:] == samples_model_format.shape[-2:] else \
                       torch.nn.functional.interpolate(mask, size=samples_model_format.shape[-2:], mode='bilinear')
        
        print(f"Selecting best sample based on LPIPS...")
        lpips_scores = compute_spatial_lpips_batch(
            model.lpips_model, clean_image, samples_model_format, mask_resized, smooth_boundary=True
        )
        
        best_idx = lpips_scores.argmax()
        best_score = lpips_scores[best_idx].item()
        best_sample = samples_model_format[best_idx].clone()
        
        print(f"Best sample selected: #{best_idx+1} (LPIPS: {best_score:.4f})")
        
        del all_samples, samples_model_format, lpips_scores
        return best_sample, best_score

    def _run_tiled(model, clean_image, mask, prompt,
                   num_samples, steps, guidance_scale, batch_size,
                   seed_offset, base, overlap):
        """Genera un crop panorámico dividiéndolo en tiles de `base`x`base`
        con solape y fusionando el resultado con blending lineal.
        Devuelve (imagen_fusionada_en_[-1,1], lpips_score_global)."""
        _, _, H, W = clean_image.shape
        device = clean_image.device
        dtype = clean_image.dtype

        # El eje corto puede ser < base (p. ej. lata 600x348): padear imagen y
        # máscara al cuadrado base para que cada tile sea exactamente base x base.
        padH = max(H, base)
        padW = max(W, base)
        pad_top = (padH - H) // 2
        pad_left = (padW - W) // 2
        padded_img = torch.zeros(1, 3, padH, padW, device=device, dtype=dtype)
        padded_img[:, :, pad_top:pad_top + H, pad_left:pad_left + W] = clean_image
        padded_mask = torch.zeros(1, 1, padH, padW, device=device, dtype=dtype)
        padded_mask[:, :, pad_top:pad_top + H, pad_left:pad_left + W] = mask

        # Orientación: panorámica horizontal o vertical
        horizontal = padW >= padH
        long_side = padW if horizontal else padH

        # Posiciones de los tiles a lo largo del eje largo (con solape)
        if base >= long_side:
            positions = [0]
        else:
            step = max(1, base - overlap)
            positions = list(range(0, long_side - base + 1, step))
            if positions[-1] + base < long_side:
                positions.append(long_side - base)
            positions = sorted(set(max(0, p) for p in positions))

        tile_results = []  # (imagen tile en [-1,1] 1x3xbasexbase, offset)

        for pos in positions:
            if horizontal:
                img_tile = padded_img[:, :, :, pos:pos + base]
                mask_tile = padded_mask[:, :, :, pos:pos + base]
            else:
                img_tile = padded_img[:, :, pos:pos + base, :]
                mask_tile = padded_mask[:, :, pos:pos + base, :]

            print(f"  [tiling] tile en pos={pos}: {img_tile.shape[-2]}x{img_tile.shape[-1]}")
            best, _ = _run_tile_selection(model, img_tile, mask_tile, prompt,
                                          num_samples, steps, guidance_scale,
                                          batch_size, seed_offset + pos)
            tile_results.append((best, pos))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ====== Fusión con blending lineal (media ponderada en solapes) ======
        fused = torch.zeros(1, 3, padH, padW, device=device, dtype=dtype)
        weight = torch.zeros(1, 1, padH, padW, device=device, dtype=dtype)

        for img_tile, pos in tile_results:
            if horizontal:
                dest = min(base, padW - pos)
                fused[:, :, :, pos:pos + dest] += img_tile[:, :, :, :dest]
                weight[:, :, :, pos:pos + dest] += 1.0
            else:
                dest = min(base, padH - pos)
                fused[:, :, pos:pos + dest, :] += img_tile[:, :, :dest, :]
                weight[:, :, pos:pos + dest, :] += 1.0

        fused_padded = fused / weight.clamp(min=1.0)
        # Recortar el relleno para devolver la imagen a su tamaño natural
        fused = fused_padded[:, :, pad_top:pad_top + H, pad_left:pad_left + W].contiguous()

        # ====== LPIPS global sobre la imagen fusionada ======
        mask_resized = torch.nn.functional.interpolate(
            mask, size=(H, W), mode='bilinear')
        lpips_scores = compute_spatial_lpips_batch(
            model.lpips_model, clean_image, fused, mask_resized, smooth_boundary=True
        )
        best_score = float(lpips_scores.max().item())
        print(f"Tiling fusionado: LPIPS global={best_score:.4f}")
        del tile_results
        return fused, best_score

    def _run_tile_selection(model, img_tile, mask_tile, prompt,
                            num_samples, steps, guidance_scale, batch_size, seed):
        """Genera `num_samples` candidatos para un tile y devuelve el mejor
        por LPIPS (misma lógica que el caso simple pero sin resize previo)."""
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            generator = torch.Generator(device=img_tile.device).manual_seed(start_idx + seed)
            b_clean = img_tile.repeat(current_batch_size, 1, 1, 1)
            b_mask = mask_tile.repeat(current_batch_size, 1, 1, 1)
            b_samples = model.generate(
                image=b_clean, mask=b_mask, prompt=prompt,
                num_inference_steps=steps, guidance_scale=guidance_scale,
                generator=generator,
            )
            all_samples.append((b_samples * 2.0) - 1.0)
        samples = torch.cat(all_samples, dim=0)
        mask_resized = torch.nn.functional.interpolate(
            mask_tile, size=samples.shape[-2:], mode='bilinear')
        lpips = compute_spatial_lpips_batch(model.lpips_model, img_tile, samples, mask_resized, smooth_boundary=True)
        best_idx = int(lpips.argmax())
        return samples[best_idx].unsqueeze(0), float(lpips[best_idx].item())



    # Transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 4
    os.makedirs(args.output_dir, exist_ok=True)
    
    inference_log = {
        "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "checkpoint": args.checkpoint,
        "object_class": args.object_class,
        "class_name": args.class_name,
        "defect_type": args.defect_type,
        "results": []
    }
    
    # Mode A: Dynamic Dataset Generation
    if args.total_images > 0 and args.data_dir and args.defect_type:
        print(f"\n{'='*60}\nDynamic Generation Mode Activated\n{'='*60}")
        num_good, num_masks, good_dir, mask_dir = count_available_resources(args.data_dir, args.object_class, args.defect_type)
        
        if num_good == 0 or num_masks == 0:
            print("Error: Missing images or masks.")
            return
            
        generation_plan = calculate_generation_plan(num_good, num_masks, args.total_images)
        good_files = sorted([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        defect_output_dir = os.path.join(args.output_dir, args.defect_type)
        os.makedirs(defect_output_dir, exist_ok=True)
        
        write_status(
            args.output_dir, state="running", step=0, total_steps=len(generation_plan),
            object_class=args.object_class, defect_type=args.defect_type, images=[]
        )
        generated_previews = []

        for good_idx, mask_idx, output_idx in tqdm(generation_plan, desc=f"Generating {args.defect_type}"):
            good_path = os.path.join(good_dir, good_files[good_idx])
            mask_path = os.path.join(mask_dir, mask_files[mask_idx])
            
            print(f"\n[{output_idx+1}/{len(generation_plan)}] Processing: {good_files[good_idx]}")
            
            # --- SMART CROP LOGIC START ---
            
            # Load Images as Numpy Arrays (for Smart Crop)
            # Use PIL and convert to numpy to ensure RGB format is consistent
            image_pil = Image.open(good_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            # La máscara de referencia puede tener otro tamaño que la imagen
            # buena: alinearla antes del smart crop (igual que en manual).
            if mask_pil.size != image_pil.size:
                print(f"[Info] Redimensionando máscara {mask_pil.size} -> {image_pil.size}")
                mask_pil = mask_pil.resize(image_pil.size, Image.Resampling.NEAREST)

            image_np = np.array(image_pil)
            mask_np = np.array(mask_pil)
            
            # --- DILATION LOGIC (Must match training) ---
            if args.dilate_mask:
                # Ensure kernel size is odd
                k_size = args.mask_kernel_size if args.mask_kernel_size % 2 == 1 else args.mask_kernel_size + 1
                kernel = np.ones((k_size, k_size), np.uint8)
                
                # Apply dilation
                # Note: mask_np is usually 0-255. cv2.dilate works fine on uint8.
                mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                
                print(f"Dilated mask with kernel {k_size}")
            # --------------------------------------------------

            # Sin recorte: se redimensiona la imagen COMPLETA a 512x512
            # (igual que el pipeline de test en entrenamiento), se genera
            # a 512 y luego se restaura a la resolución original para el
            # trío y la fusión en el HMI.
            orig_h, orig_w = image_np.shape[:2]
            if (orig_h, orig_w) != (512, 512):
                img_interp = cv2.INTER_AREA if (orig_h > 512 or orig_w > 512) else cv2.INTER_CUBIC
                crop_img_np = cv2.resize(image_np, (512, 512), interpolation=img_interp)
                crop_mask_np = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
            else:
                crop_img_np = image_np.copy()
                crop_mask_np = mask_np.copy()
            
            # Convert to Tensor
            
            # Image: [0, 255] -> [0.0, 1.0] -> Normalize to [-1.0, 1.0]
            # transforms.ToTensor() handles the HWC->CHW and /255 division automatically
            img_tensor = transforms.ToTensor()(crop_img_np)
            img_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).to(device, dtype=dtype)
            
            # Mask: [0, 255] -> [0.0, 1.0]
            mask_tensor = transforms.ToTensor()(crop_mask_np).unsqueeze(0).to(device, dtype=dtype)
            
            
            with torch.no_grad():
                defect_img, lpips_score = fixed_inference_batch(
                    model, img_tensor, mask_tensor, args.object_class, args.defect_type, 
                    num_samples=args.num_samples, steps=args.steps, guidance_scale=args.guidance_scale, batch_size=batch_size,
                    custom_prompt=args.prompt,
                )
            
            # Model works at 512; restore output to the original input resolution
            # (input size == output size, automatic for any uploaded tile size).
            defect_img_final = torch.nn.functional.interpolate(
                defect_img.unsqueeze(0), size=(orig_h, orig_w), mode='area'
            ).squeeze(0)
            mask_tensor_final = torch.nn.functional.interpolate(
                mask_tensor, size=(orig_h, orig_w), mode='nearest'
            ).squeeze(0)
            img_tensor_final = torch.nn.functional.interpolate(
                img_tensor, size=(orig_h, orig_w), mode='area'
            ).squeeze(0)
            
            # Save generated image
            # El nombre incluye el stem de la imagen original para poder ubicar
            # de qué buena procede (p. ej. Good_..._crop_1_gen0000_generated.png).
            good_basename = os.path.splitext(good_files[good_idx])[0]
            output_name = f"{good_basename}_gen{output_idx:04d}_generated.png"
            output_path = os.path.join(defect_output_dir, output_name)
            save_image((defect_img_final.float() + 1) / 2, output_path)
            
            # Save mask and original (at the original input resolution)
            mask_name = f"{good_basename}_gen{output_idx:04d}_mask.png"
            orig_name = f"{good_basename}_gen{output_idx:04d}_original.png"
            save_image(mask_tensor_final.float(), os.path.join(defect_output_dir, mask_name))
            save_image((img_tensor_final.float() + 1) / 2, os.path.join(defect_output_dir, orig_name))

            inference_log["results"].append({
                "output_idx": output_idx, "input_image": good_path, "lpips_score": lpips_score
            })

            generated_previews.append({
                "output_idx": output_idx,
                "generated": os.path.join(args.defect_type, output_name),
                "mask": os.path.join(args.defect_type, mask_name),
                "original": os.path.join(args.defect_type, orig_name),
                "lpips_score": lpips_score,
            })
            write_status(
                args.output_dir, state="running", step=output_idx + 1, total_steps=len(generation_plan),
                object_class=args.object_class, defect_type=args.defect_type,
                images=generated_previews[-12:],
            )

            torch.cuda.empty_cache()

    # Mode B: Manual Inference (single good image + user-drawn mask).
    #
    # Permite "Inferencia manual": el usuario selecciona un crop/imagen buena y
    # pinta una máscara a mano. El modelo genera defectos solo en esa zona,
    # basándose en el checkpoint entrenado. La salida es idéntica al modo
    # automático (triplet original/máscara/generado + inference_log.json), por
    # lo que los resultados aparecen en la pestaña Resultados y son fusionables
    # desde el HMI.
    elif args.image_path and args.mask_path:
        # total_images = nº de variantes con la MISMA máscara (el usuario elige la preferida)
        manual_total = int(getattr(args, "total_images", 0) or 0)
        if manual_total < 1:
            manual_total = 1
        print(f"\n{'='*60}\nManual Inference Mode (single image + user mask) x{manual_total}\n{'='*60}")
        write_status(
            args.output_dir, state="running", step=0, total_steps=manual_total,
            object_class=args.object_class, defect_type=args.defect_type or "", images=[]
        )

        image_pil = Image.open(args.image_path).convert("RGB")
        mask_pil = Image.open(args.mask_path).convert("L")

        # Alinear la máscara a la imagen si las dimensiones no coinciden.
        if mask_pil.size != image_pil.size:
            print(f"[Info] Redimensionando máscara {mask_pil.size} -> {image_pil.size}")
            mask_pil = mask_pil.resize(image_pil.size, Image.Resampling.NEAREST)

        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)

        non_zero = int((mask_np > 0).sum())
        if non_zero == 0:
            print("Error: la máscara está vacía. Pinta la zona donde quieres generar el defecto.")
            write_status(args.output_dir, state="failed", error="La máscara está vacía.")
            return

        # --- DILATION LOGIC (Must match training) ---
        if args.dilate_mask:
            k_size = args.mask_kernel_size if args.mask_kernel_size % 2 == 1 else args.mask_kernel_size + 1
            kernel = np.ones((k_size, k_size), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            print(f"Dilated mask with kernel {k_size}")

        # Sin recorte: se redimensiona la imagen COMPLETA a 512x512
        # (igual que en entrenamiento/test), se genera a 512 y luego se
        # restaura a la resolución original para el trío y la fusión.
        orig_h, orig_w = image_np.shape[:2]
        if (orig_h, orig_w) != (512, 512):
            img_interp = cv2.INTER_AREA if (orig_h > 512 or orig_w > 512) else cv2.INTER_CUBIC
            crop_img_np = cv2.resize(image_np, (512, 512), interpolation=img_interp)
            crop_mask_np = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            crop_img_np = image_np.copy()
            crop_mask_np = mask_np.copy()

        # Guardar la máscara recortada al MISMO encuadre que original/generado, para que
        # el trío de resultados sea coherente (misma zona y tamaño, defecto en la misma
        # posición). Si no se recortó (crop == imagen) es idéntica a lo pintado.
        if crop_mask_np.shape[:2] != (orig_h, orig_w):
            mask_for_save_np = cv2.resize(crop_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_for_save_np = crop_mask_np.copy()
        # Convertir a tensor para save_image (0-1)
        mask_for_save_tensor = torch.from_numpy(mask_for_save_np.astype("float32")).float() / 255.0
        # save_image espera [C,H,W] o [H,W]; lo convertimos a [1,H,W]
        if mask_for_save_tensor.dim() == 2:
            mask_for_save_tensor = mask_for_save_tensor.unsqueeze(0)

        img_tensor = transforms.ToTensor()(crop_img_np)
        img_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(device, dtype=dtype)

        mask_tensor = transforms.ToTensor()(crop_mask_np).unsqueeze(0).to(device, dtype=dtype)

        # Imagen original a resolución del crop (para triplet)
        img_tensor_final = torch.nn.functional.interpolate(
            img_tensor, size=(orig_h, orig_w), mode='area'
        ).squeeze(0)

        good_basename = os.path.splitext(os.path.basename(args.image_path))[0]
        generated_previews = []

        for output_idx in range(manual_total):
            print(f"\n[{output_idx+1}/{manual_total}] Generando variante con la máscara pintada...")
            with torch.no_grad():
                defect_img, lpips_score = fixed_inference_batch(
                    model, img_tensor, mask_tensor, args.object_class, args.defect_type,
                    num_samples=args.num_samples, steps=args.steps, guidance_scale=args.guidance_scale, batch_size=batch_size,
                    custom_prompt=args.prompt, seed_offset=output_idx * 10000,
                )

            # Model works at 512; restore output to the original input resolution.
            defect_img_final = torch.nn.functional.interpolate(
                defect_img.unsqueeze(0), size=(orig_h, orig_w), mode='area'
            ).squeeze(0)

            output_name = f"{good_basename}_gen{output_idx:04d}_generated.png"
            mask_name = f"{good_basename}_gen{output_idx:04d}_mask.png"
            orig_name = f"{good_basename}_gen{output_idx:04d}_original.png"

            save_image((defect_img_final.float() + 1) / 2, os.path.join(args.output_dir, output_name))
            # Máscara exacta dibujada por el usuario (sin artefacto de smart_crop/resize)
            save_image(mask_for_save_tensor.float(), os.path.join(args.output_dir, mask_name))
            save_image((img_tensor_final.float() + 1) / 2, os.path.join(args.output_dir, orig_name))

            inference_log["results"].append({
                "output_idx": output_idx, "input_image": args.image_path, "lpips_score": lpips_score
            })

            generated_previews.append({
                "output_idx": output_idx,
                "generated": output_name,
                "mask": mask_name,
                "original": orig_name,
                "lpips_score": lpips_score,
            })
            write_status(
                args.output_dir, state="running", step=output_idx + 1, total_steps=manual_total,
                object_class=args.object_class, defect_type=args.defect_type or "",
                images=generated_previews[-12:],
            )
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    elif args.image_path:
        print("Error: --mask_path is required for single-image manual inference.")
        return
    elif args.image_dir:
        print("Error: --image_dir is not implemented. Use --image_path with --mask_path.")
        return

    # Save Log
    log_path = os.path.join(args.output_dir, "inference_log.json")
    with open(log_path, "w") as f:
        json.dump(inference_log, f, indent=4)
    print(f"\nInference log saved to: {log_path}")

    write_status(
        args.output_dir, state="completed",
        step=len(inference_log["results"]), total_steps=len(inference_log["results"]),
        object_class=args.object_class, defect_type=args.defect_type,
        images=inference_log["results"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with DefectFill model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--object_class", type=str, required=True, help="Object class")
    parser.add_argument("--class_name", type=str, default=None, help="Human-readable class name for prompts (defaults to --object_class)")
    parser.add_argument("--defect_type", type=str, help="Defect type (e.g., 'cracks')")
    parser.add_argument("--data_dir", type=str, help="Dataset root for dynamic generation")
    parser.add_argument("--image_path", type=str, help="Single image path")
    parser.add_argument("--mask_path", type=str, help="Mask path for single-image manual inference (same size as --image_path)")
    parser.add_argument("--num_samples", type=int, default=8, help="Samples per image (for LPIPS selection)")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--total_images", type=int, default=100, help="Total synthetic images to create")
    parser.add_argument("--batch_size", type=int, default=4, help="Parallel generation batch size")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--use_compile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--prompt", type=str, default=None, help="Custom generation prompt. Must contain the learned <defect> token (auto-appended if missing). English is recommended (CLIP is English-trained).")
    parser.add_argument("--dilate_mask", type=str, default="False", help="Whether to dilate masks (True/False)")
    parser.add_argument("--mask_kernel_size", type=int, default=3, help="Size of dilation kernel")
    
    args = parser.parse_args()
    args.dilate_mask = args.dilate_mask.lower() == "true" # Handle boolean conversion
    if not args.class_name:
        args.class_name = args.object_class

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        inference(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        write_status(args.output_dir, state="failed", error=str(e))
        raise
