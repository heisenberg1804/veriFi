"""Download pre-trained model weights for VeriFi."""
import subprocess
from pathlib import Path

WEIGHTS_DIR = Path("data/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "clip_deepfake_torchscript": {
        "url": "https://github.com/yermandy/deepfake-detection/releases/download/v1.0/model.torchscript",
        "filename": "clip_vit_l14_deepfake.torchscript",
        "description": "CLIP ViT-L/14 LN-tuned (WACV 2026)",
    },
}

MANUAL_DOWNLOADS = {
    "df40_clip_weights": {
        "source": "https://github.com/YZY-stack/DF40",
        "instructions": "Download from Google Drive link in DF40 repo README",
        "save_as": "data/weights/clip_df40.pth",
    },
    "deepfakebench_effnet": {
        "source": "https://github.com/SCLBD/DeepfakeBench",
        "instructions": "Follow DeepfakeBench setup to get EfficientNet-B4 weights",
        "save_as": "data/weights/efficientnet_b4_ff.pth",
    },
}


def download(name: str, info: dict) -> bool:
    dest = WEIGHTS_DIR / info["filename"]
    if dest.exists():
        print(f"  [skip] {name}: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"  [download] {name}: {info['description']}")
    print(f"             {info['url']}")
    try:
        subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", str(dest), info["url"]],
            check=True,
        )
        print(f"  [done] saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  [error] Failed to download {name}")
        return False


def main():
    print("=" * 60)
    print("VeriFi — Model Weight Downloader")
    print("=" * 60)

    print("\n── Auto-downloads ──")
    for name, info in WEIGHTS.items():
        download(name, info)

    print("\n── Manual downloads required ──")
    for name, info in MANUAL_DOWNLOADS.items():
        path = Path(info["save_as"])
        if path.exists():
            print(f"  [skip] {name}: already at {path}")
        else:
            print(f"  [TODO] {name}")
            print(f"         Source: {info['source']}")
            print(f"         {info['instructions']}")
            print(f"         Save to: {info['save_as']}")

    print("\n── Verification ──")
    for f in WEIGHTS_DIR.iterdir():
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
