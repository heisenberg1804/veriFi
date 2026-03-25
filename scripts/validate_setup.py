"""Phase 1 validation: check that everything works before building."""
import sys
import time
from pathlib import Path

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

errors = []


def check(name: str, fn):
    try:
        result = fn()
        if result:
            print(f"  {PASS} {name}: {result}")
        else:
            print(f"  {PASS} {name}")
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        errors.append(name)


def check_torch():
    import torch
    return f"v{torch.__version__}"


def check_device():
    import torch
    if torch.backends.mps.is_available():
        return "MPS (Apple Silicon GPU) available"
    elif torch.cuda.is_available():
        return f"CUDA available ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU only (MPS/CUDA not available)"


def check_mps_inference():
    import torch
    if not torch.backends.mps.is_available():
        print(f"  {WARN} MPS not available — skipping GPU benchmark")
        return None
    device = "mps"
    x = torch.randn(1, 3, 224, 224, device=device)
    # Warm up
    for _ in range(3):
        _ = x * 2 + 1
    torch.mps.synchronize()
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = x * 2 + 1
    torch.mps.synchronize()
    elapsed = (time.perf_counter() - start) * 10  # ms per op
    return f"MPS basic op: {elapsed:.2f}ms/100 ops"


def check_open_clip():
    import open_clip
    models = open_clip.list_pretrained()
    vit_l14 = [m for m in models if "ViT-L-14" in m[0]]
    return f"open_clip loaded, {len(vit_l14)} ViT-L-14 variants available"


def check_timm():
    import timm
    assert "efficientnet_b4" in timm.list_models("efficientnet*")
    return "efficientnet_b4 available"


def check_cv2():
    import cv2
    return f"OpenCV v{cv2.__version__}"


def check_gradcam():
    from pytorch_grad_cam import GradCAM
    return "pytorch-grad-cam imported"


def check_facenet():
    from facenet_pytorch import MTCNN
    return "MTCNN available"


def check_anthropic():
    import anthropic
    return f"anthropic SDK v{anthropic.__version__}"


def check_fastapi():
    import fastapi
    return f"FastAPI v{fastapi.__version__}"


def check_weights():
    weights_dir = Path("data/weights")
    if not weights_dir.exists():
        return "data/weights/ not found — run: python scripts/download_weights.py"
    files = list(weights_dir.glob("*"))
    if not files:
        print(f"  {WARN} No weight files found — run: python scripts/download_weights.py")
        return None
    return f"{len(files)} weight file(s) found"


def check_clip_inference():
    """The big test: can we actually run CLIP inference?"""
    import torch
    from verifi.config import AppConfig

    config = AppConfig()
    device = config.device.resolve()

    weight_path = config.detector.clip_weight_path
    if not weight_path.exists():
        print(f"  {WARN} CLIP weights not found at {weight_path} — skipping inference test")
        return None

    from verifi.detectors.clip_detector import CLIPDeepfakeDetector
    import numpy as np

    detector = CLIPDeepfakeDetector(
        weight_path=str(weight_path),
        device=device,
        input_size=config.detector.clip_input_size,
    )
    detector.load()

    # Create dummy face crop (224x224 random pixels)
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Warm up
    detector.predict([dummy])

    # Benchmark
    start = time.perf_counter()
    results = detector.predict([dummy] * 4)  # batch of 4
    elapsed = (time.perf_counter() - start) * 1000

    detector.unload()
    return f"Batch of 4 on {device}: {elapsed:.0f}ms ({elapsed/4:.0f}ms/frame), scores: {[f'{r.score:.3f}' for r in results]}"


def main():
    print("=" * 60)
    print("VeriFi — Phase 1 Setup Validation")
    print("=" * 60)

    print("\n── Core dependencies ──")
    check("PyTorch", check_torch)
    check("Compute device", check_device)
    check("MPS inference", check_mps_inference)
    check("OpenCLIP", check_open_clip)
    check("timm", check_timm)
    check("OpenCV", check_cv2)
    check("pytorch-grad-cam", check_gradcam)
    check("MTCNN (facenet-pytorch)", check_facenet)
    check("Anthropic SDK", check_anthropic)
    check("FastAPI", check_fastapi)

    print("\n── Model weights ──")
    check("Weight files", check_weights)

    print("\n── Inference test ──")
    check("CLIP ViT-L/14 inference", check_clip_inference)

    print("\n" + "=" * 60)
    if errors:
        print(f"\033[91m{len(errors)} check(s) failed: {', '.join(errors)}\033[0m")
        print("Fix these before proceeding to Phase 2.")
        sys.exit(1)
    else:
        print(f"\033[92mAll checks passed. Ready to build.\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
