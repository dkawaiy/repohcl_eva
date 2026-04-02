#!/usr/bin/env python3
import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List

try:
    from PIL import Image, ImageChops, ImageOps
except Exception:
    print("FAIL: missing dependency Pillow. Please install with: pip install Pillow")
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "target_projects" / "thumbnailator"
INPUT_IMAGE = PROJECT_DIR / "src" / "main" / "resources" / "input.jpg"
OUTPUT_CANDIDATES = [
    PROJECT_DIR / "output.png",
    PROJECT_DIR / "output" / "output.png",
]
CONTAINER_NAME = "test_env_demo1"
CONTAINER_PROJECT_DIR = "/root/target_projects/thumbnailator"


def run_maven() -> Tuple[bool, str]:
    cmd = [
        "docker",
        "exec",
        "-i",
        CONTAINER_NAME,
        "bash",
        "-lc",
        f"cd {CONTAINER_PROJECT_DIR} && mvn -q compile exec:java",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0, proc.stdout


def pick_output() -> Optional[Path]:
    for p in OUTPUT_CANDIDATES:
        if p.exists() and p.is_file():
            return p
    return None


def mse_rgba(img1: Image.Image, img2: Image.Image) -> float:
    diff = ImageChops.difference(img1, img2)
    hist = diff.histogram()
    sq_sum = 0.0
    idx = 0
    for _ in range(4):
        for v in range(256):
            count = hist[idx]
            sq_sum += (v * v) * count
            idx += 1
    total_values = img1.width * img1.height * 4
    return sq_sum / max(total_values, 1)


def expected_variants(input_path: Path) -> List[Image.Image]:
    src = Image.open(input_path).convert("RGBA")
    # Thumbnailator 语义：缩放 + 中心裁剪得到 200x200。
    fitted = ImageOps.fit(src, (200, 200), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    # 对旋转方向做兼容（不同库对正角度视觉方向定义不同）。
    variants = []
    for angle in (-90, 90):
        rotated = fitted.rotate(angle, expand=False)
        alpha = rotated.split()[-1].point(lambda a: int(round(a * 0.5)))
        rotated.putalpha(alpha)
        variants.append(rotated)
    return variants


def validate_output(output_path: Path) -> Tuple[bool, str]:
    if not INPUT_IMAGE.exists():
        return False, f"input image not found: {INPUT_IMAGE}"

    try:
        out_img = Image.open(output_path)
    except Exception as exc:
        return False, f"cannot open output image: {exc}"

    if out_img.format != "PNG":
        return False, f"output format must be PNG, got {out_img.format}"

    out_rgba = out_img.convert("RGBA")
    if out_rgba.size != (200, 200):
        return False, f"output size must be 200x200, got {out_rgba.size[0]}x{out_rgba.size[1]}"

    alpha_band = out_rgba.split()[-1]
    alpha_data = list(alpha_band.getdata())
    if not alpha_data:
        return False, "alpha channel check failed: empty image data"

    alpha_min = min(alpha_data)
    alpha_max = max(alpha_data)
    alpha_avg = sum(alpha_data) / len(alpha_data)

    # 0.5 透明度通常接近 127/128；允许小范围误差以兼容实现细节。
    if not (110 <= alpha_avg <= 145):
        return (
            False,
            f"opacity check failed, expected around 0.5 alpha, got avg={alpha_avg:.2f}, min={alpha_min}, max={alpha_max}",
        )

    candidates = expected_variants(INPUT_IMAGE)
    errors = [mse_rgba(out_rgba, cand) for cand in candidates]
    best = min(errors)

    # 使用较宽容阈值，避免因插值细节差异导致误判。
    if not math.isfinite(best) or best > 220.0:
        return False, f"pixel similarity check failed, best MSE={best:.3f}"

    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge thumbnailator output")
    parser.add_argument(
        "--skip-maven",
        action="store_true",
        help="Skip maven execution and only validate existing output image",
    )
    args = parser.parse_args()

    if not PROJECT_DIR.exists():
        print(f"FAIL: project directory not found: {PROJECT_DIR}")
        return 1

    if not args.skip_maven:
        ok, logs = run_maven()
        if not ok:
            print("FAIL: maven run in container failed")
            print(f"container: {CONTAINER_NAME}")
            print(logs[-5000:])
            return 1

    output_path = pick_output()
    if output_path is None:
        print("FAIL: output.png not found in expected locations")
        print(f"checked: {OUTPUT_CANDIDATES[0]} and {OUTPUT_CANDIDATES[1]}")
        return 1

    valid, reason = validate_output(output_path)
    if not valid:
        print(f"FAIL: {reason}")
        print(f"output path: {output_path}")
        return 1

    print("SUCCESS")
    print(f"output path: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())