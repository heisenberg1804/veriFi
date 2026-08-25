"""Generate an AI-video evaluation set via Veo 3.1 and write a manifest CSV.

Requires the google-genai package:
    pip install "verifi[evalset]"

Usage:
    export GOOGLE_API_KEY=...
    python scripts/generate_eval_set.py --out data/datasets/evalset
    python scripts/generate_eval_set.py --out data/datasets/evalset --prompts prompts.csv
    python scripts/generate_eval_set.py --out data/datasets/evalset --categories sports nature

The manifest CSV has columns: path, label, generator, source, prompt
and is written to <out>/manifest.csv.  Already-generated rows are skipped
on re-run so the script is resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROMPTS: list[dict[str, str]] = [
    # Sports — content-matched to real sports clips
    {
        "category": "sports",
        "prompt": (
            "A professional soccer player dribbling past defenders"
            " on a grass pitch, stadium crowd in background,"
            " broadcast camera angle, 4K"
        ),
    },
    {
        "category": "sports",
        "prompt": (
            "A basketball player shooting a free throw in a packed"
            " arena, close-up tracking shot, evening game lighting"
        ),
    },
    {
        "category": "sports",
        "prompt": (
            "A tennis serve in slow motion, clay court,"
            " sunlit afternoon, broadcast side camera"
        ),
    },
    # Nature / animals
    {
        "category": "nature",
        "prompt": (
            "A capybara walking calmly through a grassy wetland"
            " at golden hour, handheld documentary style"
        ),
    },
    {
        "category": "nature",
        "prompt": (
            "A hummingbird hovering at a red flower, macro lens,"
            " shallow depth of field, natural daylight"
        ),
    },
    {
        "category": "nature",
        "prompt": (
            "Waves crashing on a rocky coastline at sunset,"
            " wide angle, drone perspective slowly descending"
        ),
    },
    # Urban / people
    {
        "category": "urban",
        "prompt": (
            "A street musician playing guitar on a busy European"
            " sidewalk, pedestrians walking past, overcast afternoon"
        ),
    },
    {
        "category": "urban",
        "prompt": (
            "A food vendor preparing tacos at a night market,"
            " close-up of hands and ingredients,"
            " warm artificial lighting"
        ),
    },
    {
        "category": "urban",
        "prompt": (
            "Busy Tokyo intersection at night with neon signs,"
            " high angle, light rain reflections on pavement"
        ),
    },
    # Interview / talking head
    {
        "category": "interview",
        "prompt": (
            "A middle-aged person speaking directly to camera"
            " in a studio with soft lighting, medium close-up,"
            " shallow depth of field"
        ),
    },
    {
        "category": "interview",
        "prompt": (
            "A young woman being interviewed outdoors in a park,"
            " natural light, over-the-shoulder two-shot"
        ),
    },
    # Driving / dashcam
    {
        "category": "driving",
        "prompt": (
            "Dashcam footage driving through a suburban neighborhood"
            " on a sunny day, steady forward view,"
            " trees lining both sides"
        ),
    },
    {
        "category": "driving",
        "prompt": (
            "Driving through a mountain tunnel then emerging into"
            " bright daylight, forward-facing camera, smooth motion"
        ),
    },
    # Surveillance-style
    {
        "category": "surveillance",
        "prompt": (
            "Security camera view of a parking lot entrance,"
            " fixed wide angle, a car pulling in slowly, daylight"
        ),
    },
    {
        "category": "surveillance",
        "prompt": (
            "Overhead security camera in a retail store,"
            " a person browsing shelves,"
            " flat lighting, fixed position"
        ),
    },
]

VEO_MODEL = "veo-3.1-generate-preview"
DURATION_SECONDS = 8
POLL_INTERVAL = 30
MAX_POLL_ATTEMPTS = 60

MANIFEST_COLUMNS = ["path", "label", "generator", "source", "prompt"]


@dataclass
class GenerationResult:
    prompt: str
    category: str
    filename: str
    success: bool
    error: str = ""


@dataclass
class EvalSetGenerator:
    output_dir: Path
    api_key: str = ""
    model: str = VEO_MODEL
    duration: int = DURATION_SECONDS
    _existing: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.api_key:
            self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._existing = self._load_existing()

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.csv"

    def _load_existing(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        prompts = set()
        with open(self.manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                prompts.add(row["prompt"])
        return prompts

    def _make_filename(self, prompt: str, category: str) -> str:
        slug = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return f"ai_{category}_{slug}.mp4"

    def generate_video(
        self, prompt: str, category: str,
    ) -> GenerationResult:
        filename = self._make_filename(prompt, category)
        out_path = self.output_dir / filename

        if prompt in self._existing:
            print(f"  SKIP (already in manifest): {filename}")
            return GenerationResult(
                prompt=prompt, category=category,
                filename=filename, success=True,
            )

        if not self.api_key:
            return GenerationResult(
                prompt=prompt, category=category,
                filename=filename, success=False,
                error="GOOGLE_API_KEY not set",
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return GenerationResult(
                prompt=prompt, category=category,
                filename=filename, success=False,
                error="google-genai not installed",
            )

        client = genai.Client(api_key=self.api_key)
        try:
            operation = client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=types.GenerateVideoConfig(
                    number_of_videos=1,
                    duration_seconds=self.duration,
                    aspect_ratio="16:9",
                ),
            )
        except Exception as e:
            return GenerationResult(
                prompt=prompt, category=category,
                filename=filename, success=False,
                error=f"API submission failed: {e}",
            )

        for attempt in range(MAX_POLL_ATTEMPTS):
            if operation.done:
                break
            time.sleep(POLL_INTERVAL)
            try:
                operation = client.operations.get(operation)
            except Exception as e:
                return GenerationResult(
                    prompt=prompt, category=category,
                    filename=filename, success=False,
                    error=f"Poll failed at attempt {attempt}: {e}",
                )
        else:
            timeout = MAX_POLL_ATTEMPTS * POLL_INTERVAL
            return GenerationResult(
                prompt=prompt, category=category,
                filename=filename, success=False,
                error=f"Timed out after {timeout}s",
            )

        if (
            not operation.result
            or not operation.result.generated_videos
        ):
            return GenerationResult(
                prompt=prompt, category=category,
                filename=filename, success=False,
                error="No videos in response",
            )

        video = operation.result.generated_videos[0]
        try:
            client.files.download(
                file=video.video, download_path=str(out_path),
            )
        except Exception as e:
            return GenerationResult(
                prompt=prompt, category=category,
                filename=filename, success=False,
                error=f"Download failed: {e}",
            )

        self._append_manifest(filename, prompt)
        self._existing.add(prompt)
        return GenerationResult(
            prompt=prompt, category=category,
            filename=filename, success=True,
        )

    def _append_manifest(self, filename: str, prompt: str) -> None:
        write_header = not self.manifest_path.exists()
        with open(self.manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "path": filename,
                "label": "ai",
                "generator": self.model,
                "source": "generated",
                "prompt": prompt,
            })

    def run(
        self, prompts: list[dict[str, str]],
    ) -> list[GenerationResult]:
        results = []
        total = len(prompts)
        for i, entry in enumerate(prompts, 1):
            prompt = entry["prompt"]
            category = entry.get("category", "unknown")
            print(f"[{i}/{total}] {category}: {prompt[:80]}...")
            result = self.generate_video(prompt, category)
            if result.success:
                print(f"  OK: {result.filename}")
            else:
                print(f"  FAIL: {result.error}")
            results.append(result)
        return results


def load_prompts_from_csv(path: str) -> list[dict[str, str]]:
    prompts = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "prompt" not in row:
                raise ValueError(
                    "Prompt CSV must have a 'prompt' column,"
                    f" got: {list(row.keys())}"
                )
            prompts.append({
                "prompt": row["prompt"],
                "category": row.get("category", "unknown"),
            })
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate AI video evaluation set via Veo 3.1",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for videos + manifest.csv",
    )
    parser.add_argument(
        "--prompts",
        help="CSV with 'prompt' and optional 'category' columns",
    )
    parser.add_argument(
        "--categories", nargs="+",
        help="Filter default prompts to these categories",
    )
    parser.add_argument(
        "--model", default=VEO_MODEL,
        help=f"Veo model ID (default: {VEO_MODEL})",
    )
    parser.add_argument(
        "--duration", type=int, default=DURATION_SECONDS,
        help="Video duration in seconds",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without generating",
    )
    args = parser.parse_args()

    if args.prompts:
        prompts = load_prompts_from_csv(args.prompts)
    else:
        prompts = DEFAULT_PROMPTS
        if args.categories:
            cats = set(args.categories)
            prompts = [
                p for p in prompts if p["category"] in cats
            ]

    if not prompts:
        print("No prompts to generate.", file=sys.stderr)
        sys.exit(1)

    print(f"Eval set: {len(prompts)} prompts -> {args.out}")
    if args.dry_run:
        for p in prompts:
            print(f"  [{p.get('category', '?')}] {p['prompt']}")
        return

    generator = EvalSetGenerator(
        output_dir=args.out,
        model=args.model,
        duration=args.duration,
    )
    results = generator.run(prompts)

    ok = sum(1 for r in results if r.success)
    fail = sum(1 for r in results if not r.success)
    print(f"\nDone: {ok} succeeded, {fail} failed")
    if generator.manifest_path.exists():
        print(f"Manifest: {generator.manifest_path}")


if __name__ == "__main__":
    main()
