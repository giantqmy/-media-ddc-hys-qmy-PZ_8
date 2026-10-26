"""Save PCDNet validation predictions on original S0 color images.

Example (run from PZ_8):
    python visualize_pcdnet_val.py --weights runs/train/pcdnet_s0_aolp_dolp/weights/best.pt
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PCDNET_ROOT = ROOT.parents[1] / "compare" / "AAAI24-PCDNet"
COLORS = ((45, 190, 70), (210, 180, 40), (0, 140, 255), (180, 80, 230), (255, 160, 40), (40, 210, 220))

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="PCDNet best.pt or last.pt from train_pcdnet.py")
    parser.add_argument("--pcdnet-root", type=Path, default=DEFAULT_PCDNET_ROOT)
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz6.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/val/pcdnet_visualization")
    parser.add_argument("--imgsz", type=int, default=512, help="use the training image size")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0", help="CUDA device index, or cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for saved predictions")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--half", action="store_true", help="FP16 inference on CUDA")
    parser.add_argument("--save-txt", action="store_true", help="save class cx cy w h confidence (normalized to original image)")
    parser.add_argument("--show-gt", action="store_true", help="also save separate ground-truth images in gt/")
    parser.add_argument("--max-images", type=int, default=0, help="0 saves the entire val split")
    parser.add_argument("--line-width", type=int, default=2)
    args = parser.parse_args()
    if min(args.imgsz, args.batch, args.line_width) < 1 or args.workers < 0 or args.max_images < 0:
        parser.error("imgsz, batch and line-width must be positive; workers and max-images must be nonnegative")
    if not 0 < args.conf < 1 or not 0 <= args.iou <= 1:
        parser.error("conf must be in (0, 1) and iou in [0, 1]")
    if args.imgsz % 32:
        parser.error("imgsz must be a multiple of 32")
    return args


def restore_boxes(boxes, original_shape, ratio_pad):
    """Undo this loader's rounded resize and letterbox using separate x/y gains."""
    boxes = boxes.copy()
    (gain_y, gain_x), (pad_x, pad_y) = ratio_pad
    height, width = original_shape
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / gain_x).clip(0, width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / gain_y).clip(0, height)
    return boxes


def read_s0_image(image_path):
    import cv2
    import numpy as np

    # Use precisely the same S0 file naming convention as getPolarizationImages.
    path = Path(image_path)
    path = path.with_name(f"{path.stem}_S0_rgb.png")
    if not path.is_file():
        raise FileNotFoundError(f"S0 display image not found: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read S0 display image: {path}")
    return image


def draw_boxes(image, boxes, classes, names, line_width, confidences=None):
    import cv2
    # from ultralytics.utils.plotting import colors

    canvas = image.copy()
    # colors = ((56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72))
    scale = max(0.4, min(image.shape[:2]) / 1200)
    for index, (box, cls) in enumerate(zip(boxes, classes)):
        cls = int(cls)
        # color = colors[cls % len(colors)]
        # color = colors(cls, True)
        color = COLORS[cls % len(COLORS)]
        name = str(names[cls])
        text = f"{name} {confidences[index]:.2f}" if confidences is not None else f"GT {name}"
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        text_y = max(text_h + baseline + 4, y1)
        cv2.rectangle(canvas, (x1, text_y - text_h - baseline - 4), (x1 + text_w + 4, text_y), color, -1)
        cv2.putText(canvas, text, (x1 + 2, text_y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_image(path, image):
    import cv2

    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Failed to encode visualization: {path}")
    encoded.tofile(str(path))


def visualize(args):
    import numpy as np
    import torch

    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.files import increment_path

    from pcdnet_trainer.trainer import (
        PCDNetTrainer,
        _import_pcdnet,
        _make_loader,
        _reset_inference_grids,
        _torch_load,
        _unwrap,
        split_polarization_inputs,
    )

    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    device = PCDNetTrainer._select_device(args.device)
    _, _, non_max_suppression, xywh2xyxy, _ = _import_pcdnet(args.pcdnet_root)
    checkpoint = _torch_load(args.weights, torch.device("cpu"))
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a full PCDNet checkpoint dictionary containing model or ema")
    model = next((checkpoint[key] for key in ("ema", "model")
                  if isinstance(checkpoint.get(key), torch.nn.Module)), None)
    if model is None:
        raise ValueError("A state_dict-only file is unsupported; use best.pt or last.pt saved by train_pcdnet.py")
    model = _unwrap(model).float().to(device).eval()
    _reset_inference_grids(model)
    data = check_det_dataset(args.data, autodownload=False)
    if not data.get("val"):
        raise ValueError(f"No val split configured in {args.data}")
    if model.model[-1].nc != data["nc"]:
        raise ValueError(f"Checkpoint has {model.model[-1].nc} classes but dataset has {data['nc']}; use matching trained weights/data")
    # Do not call the training loader's head-reset function: inference must preserve all weights.
    half = args.half and device.type == "cuda"
    if half:
        model.half()
    elif args.half:
        print("FP16 requires CUDA; using FP32 on CPU")
    stride = max(int(model.stride.max()), 32)
    if args.imgsz % stride:
        raise ValueError(f"imgsz must be a multiple of model stride {stride}")
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "cache": False, "rect": False})
    loader = _make_loader(data, cfg, "val", args.batch, args.workers, augment=False)
    output = increment_path(args.output, mkdir=True)
    for folder in ("images", *(('labels',) if args.save_txt else ()), *(('gt',) if args.show_gt else ())):
        (output / folder).mkdir(parents=True, exist_ok=True)
    total = len(loader.dataset)
    if args.max_images:
        total = min(total, args.max_images)
    processed = detection_count = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as records, torch.inference_mode():
        for batch in loader:
            if processed >= total:
                break
            count = min(len(batch["im_file"]), total - processed)
            images = batch["img"][:count].to(device, non_blocking=True)
            images = (images.half() if half else images.float()) / 255.0
            predictions, _ = model(split_polarization_inputs(images))
            detections = non_max_suppression(predictions, conf_thres=args.conf, iou_thres=args.iou, multi_label=True)
            for index, detection in enumerate(detections):
                source = batch["im_file"][index]
                original_shape = batch["ori_shape"][index]
                ratio_pad = batch["ratio_pad"][index]
                image = read_s0_image(source)
                if tuple(image.shape[:2]) != tuple(original_shape):
                    raise ValueError(f"S0 image shape differs from loader metadata: {source}")
                prediction = detection.detach().float().cpu().numpy()
                prediction[:, :4] = restore_boxes(prediction[:, :4], original_shape, ratio_pad)
                stem = f"{processed + 1:06d}_{Path(source).stem}"
                filename = f"{stem}.jpg"
                canvas = draw_boxes(image, prediction[:, :4], prediction[:, 5], data["names"],
                                    args.line_width, prediction[:, 4])
                save_image(output / "images" / filename, canvas)
                if args.show_gt:
                    mask = batch["batch_idx"].view(-1) == index
                    labels = batch["cls"][mask].view(-1).cpu().numpy()
                    boxes = batch["bboxes"][mask].clone()
                    height, width = images.shape[2:]
                    boxes *= boxes.new_tensor([width, height, width, height])
                    boxes = restore_boxes(xywh2xyxy(boxes).cpu().numpy(), original_shape, ratio_pad)
                    save_image(output / "gt" / filename,
                               draw_boxes(image, boxes, labels, data["names"], args.line_width))
                if args.save_txt:
                    height, width = original_shape
                    lines = []
                    for x1, y1, x2, y2, confidence, cls in prediction:
                        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                                  (x2 - x1) / width, (y2 - y1) / height, confidence)
                        lines.append(f"{int(cls)} " + " ".join(f"{value:.6g}" for value in values))
                    (output / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                records.write(json.dumps({"source": str(source), "image": f"images/{filename}",
                                          "detections": prediction.tolist()}, ensure_ascii=False) + "\n")
                processed += 1
                detection_count += len(prediction)
                print(f"[{processed}/{total}] {Path(source).name}: {len(prediction)} detections")
    summary = {"weights": str(args.weights.resolve()), "data": str(args.data.resolve()), "split": "val",
               "images": processed, "detections": detection_count, "conf": args.conf, "iou": args.iou,
               "imgsz": args.imgsz, "names": data["names"],
               "detection_columns": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
               "coordinates": "original S0 image pixels"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {processed} images and {detection_count} detections to {output.resolve()}")


if __name__ == "__main__":
    visualize(parse_args())
