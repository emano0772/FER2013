

import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 48


class ConvBlock(nn.Module):
    """Two conv layers + BN + ReLU + MaxPool + Dropout."""
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class FER2013CNN(nn.Module):
    """
    4-block CNN for FER2013.
    Input : (B, 1, 48, 48)  — grayscale, normalised to [0, 1]
    Output: (B, 7)          — raw logits
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   64,  dropout=0.25),   # → (B,  64, 24, 24)
            ConvBlock(64,  128, dropout=0.25),   # → (B, 128, 12, 12)
            ConvBlock(128, 256, dropout=0.30),   # → (B, 256,  6,  6)
            ConvBlock(256, 512, dropout=0.30),   # → (B, 512,  3,  3)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)      # → (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)



def load_model(checkpoint_path: str, device: torch.device) -> FER2013CNN:
    """Load weights from a .pth checkpoint into FER2013CNN."""
    model = FER2013CNN(num_classes=NUM_CLASSES).to(device)
    raw = torch.load(checkpoint_path, map_location=device)

    # Accept either a raw state-dict or a wrapped dict
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state = raw['model_state_dict']
    elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        state = raw
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in '{checkpoint_path}'.\n"
            "Expected a state-dict or {{'model_state_dict': ...}}."
        )

    model.load_state_dict(state)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),   # → [0, 1] float tensor (1, 48, 48)
    ])


def predict_image(
    model:     FER2013CNN,
    transform: transforms.Compose,
    img_path:  str,
    device:    torch.device,
) -> dict:
  
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return {'file': img_path, 'error': str(e)}

    tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 48, 48)

    with torch.no_grad():
        logits = model(tensor)                        # (1, 7)
        probs  = torch.softmax(logits, dim=1)[0]      # (7,)

    pred_idx    = probs.argmax().item()
    pred_label  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx].item()
    prob_dict   = {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)}

    return {
        'file'          : img_path,
        'prediction'    : pred_label,
        'confidence'    : round(confidence, 4),
        'probabilities' : prob_dict,
    }


def predict_folder(
    model:     FER2013CNN,
    transform: transforms.Compose,
    folder:    str,
    device:    torch.device,
) -> list[dict]:
    folder_path = Path(folder)
    image_files = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"[WARNING] No images found in '{folder}'.")
        return []

    results = []
    for i, img_path in enumerate(image_files, 1):
        result = predict_image(model, transform, str(img_path), device)
        results.append(result)
        if 'error' in result:
            print(f"  [{i}/{len(image_files)}] {img_path.name}  ERROR: {result['error']}")
        else:
            print(f"  [{i}/{len(image_files)}] {img_path.name}  "
                  f"→  {result['prediction']:>10s}  ({result['confidence']*100:.1f}%)")

    return results


def save_csv(results: list[dict], output_path: str) -> None:
    fieldnames = ['file', 'prediction', 'confidence'] + CLASS_NAMES + ['error']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = {
                'file'      : r.get('file', ''),
                'prediction': r.get('prediction', ''),
                'confidence': r.get('confidence', ''),
                'error'     : r.get('error', ''),
            }
            for cls in CLASS_NAMES:
                row[cls] = r.get('probabilities', {}).get(cls, '')
            writer.writerow(row)
    print(f"\n Results saved to '{output_path}'")


def plot_single(result: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed; skipping plot.")
        return

    probs  = [result['probabilities'][c] for c in CLASS_NAMES]
    colors = ['#2ecc71' if c == result['prediction'] else '#95a5a6' for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(CLASS_NAMES, probs, color=colors, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title(
        f"Prediction: {result['prediction'].upper()}  "
        f"({result['confidence']*100:.1f}% confidence)\n"
        f"File: {Path(result['file']).name}"
    )
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{p:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='FER2013 CNN — Inference script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to saved .pth checkpoint')
    parser.add_argument('--input',      required=True,
                        help='Path to a single image OR a folder of images')
    parser.add_argument('--output',     default='results.csv',
                        help='CSV output path (folder mode only, default: results.csv)')
    parser.add_argument('--plot',       action='store_true',
                        help='Show probability bar chart (single image mode)')
    parser.add_argument('--device',     default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to run on (default: auto)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: '{args.checkpoint}'")

    print(f"Loading checkpoint : {args.checkpoint}")
    model     = load_model(args.checkpoint, device)
    transform = get_transform()
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} parameters)\n")

    # ── single image ──────────────────────────────────────────────────────────
    if os.path.isfile(args.input):
        result = predict_image(model, transform, args.input, device)
        if 'error' in result:
            sys.exit(f"[ERROR] Could not process image: {result['error']}")

        print(f"File       : {result['file']}")
        print(f"Prediction : {result['prediction'].upper()}")
        print(f"Confidence : {result['confidence']*100:.2f}%")
        print("\nAll class probabilities:")
        for cls, prob in sorted(result['probabilities'].items(),
                                key=lambda x: -x[1]):
            print(f"  {cls:>10s}  {prob:.4f}  ")

        if args.plot:
            plot_single(result)

    # ── folder ────────────────────────────────────────────────────────────────
    elif os.path.isdir(args.input):
        print(f"Scanning folder : {args.input}\n")
        results = predict_folder(model, transform, args.input, device)
        if results:
            save_csv(results, args.output)
            # Print a quick summary
            from collections import Counter
            counts = Counter(r.get('prediction', 'error') for r in results)
            print("\nPrediction summary:")
            for label, count in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {label:>12s} : {count}")

    else:
        sys.exit(f"[ERROR] --input must be a file or directory: '{args.input}'")


if __name__ == '__main__':
    main()
