"""
Phase 2 integration test: Video → Scenes → Frames → Faces → Detector scores.

Save to: scripts/test_phase2.py

Run:  python scripts/test_phase2.py data/sample_videos/YOUR_VIDEO.mp4
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main(video_path: str):
    from verifi.detectors.frequency import FrequencyAnalyzer
    from verifi.detectors.temporal import TemporalAnalyzer
    from verifi.ingestion.validator import validate_video
    from verifi.preprocessing.face_detector import FaceDetectionPipeline
    from verifi.sampling.frame_selector import select_frames
    from verifi.sampling.scene_detector import detect_scenes

    print("=" * 65)
    print("VeriFi — Phase 2 Integration Test")
    print("=" * 65)

    # ── Stage 1: Validate video ──
    print("\n── Stage 1: Video validation ──")
    t0 = time.perf_counter()
    result = validate_video(video_path)

    if not result.valid:
        print(f"  [FAIL] Validation failed: {result.errors}")
        sys.exit(1)

    meta = result.metadata
    print(f"  [OK] {meta.filename}")
    print(f"       Duration: {meta.duration_sec:.1f}s | {meta.resolution} | {meta.fps:.1f} fps")
    print(f"       Codec: {meta.codec} | Audio: {meta.has_audio} | Size: {meta.file_size_mb:.1f} MB")
    print(f"       Hash: {meta.file_hash}")
    if result.warnings:
        for w in result.warnings:
            print(f"       [WARN] {w}")
    print(f"       Time: {time.perf_counter() - t0:.2f}s")

    # ── Stage 2: Scene detection ──
    print("\n── Stage 2: Scene detection ──")
    t0 = time.perf_counter()
    scene_analysis = detect_scenes(video_path)

    print(f"  [OK] {scene_analysis.num_scenes} scene(s) detected, {len(scene_analysis.boundaries)} boundary(ies)")
    for s in scene_analysis.scenes:
        print(f"       Scene {s.scene_id}: {s.start_sec:.1f}s → {s.end_sec:.1f}s ({s.frame_count} frames)")
    print(f"       Time: {time.perf_counter() - t0:.2f}s")

    # ── Stage 3: Smart frame selection ──
    print("\n── Stage 3: Smart frame selection ──")
    t0 = time.perf_counter()
    frames = select_frames(
        video_path=video_path,
        scene_analysis=scene_analysis,
        frame_budget=30,
        transition_margin=2,
        min_laplacian_var=50.0,  # lower threshold for AI-generated video
    )

    key_count = sum(1 for f in frames if f.selection_reason == "key_frame")
    trans_count = sum(1 for f in frames if f.selection_reason == "transition")
    print(f"  [OK] {len(frames)} frames selected ({key_count} key, {trans_count} transition)")
    print(f"       Blur scores: min={min(f.blur_score for f in frames):.0f}, "
          f"max={max(f.blur_score for f in frames):.0f}, "
          f"mean={np.mean([f.blur_score for f in frames]):.0f}")
    print(f"       Time: {time.perf_counter() - t0:.2f}s")

    # ── Stage 4: Face detection ──
    print("\n── Stage 4: Face detection ──")
    t0 = time.perf_counter()
    face_pipeline = FaceDetectionPipeline(
        device="cpu",  # MTCNN is faster on CPU
        target_size=224,
        margin_ratio=0.3,
        min_confidence=0.90,  # slightly lower for AI video
    )
    face_pipeline.load()

    frame_tuples = [(f.image, f.frame_idx, f.timestamp_sec) for f in frames]
    face_results = face_pipeline.detect_batch(frame_tuples)

    frames_with_faces = sum(1 for fr in face_results if fr.has_faces)
    total_detections = sum(fr.num_faces for fr in face_results)
    unique_ids = set()
    for fr in face_results:
        for face in fr.faces:
            unique_ids.add(face.face_id)

    print(f"  [OK] {frames_with_faces}/{len(frames)} frames have faces")
    print(f"       Total detections: {total_detections}")
    print(f"       Unique face IDs: {len(unique_ids)}")
    if face_results and face_results[0].has_faces:
        f0 = face_results[0].faces[0]
        print(f"       First face: ID={f0.face_id}, conf={f0.confidence:.3f}, "
              f"bbox=({f0.bbox.x},{f0.bbox.y},{f0.bbox.w},{f0.bbox.h}), "
              f"crop={f0.crop.shape}")
    print(f"       Time: {time.perf_counter() - t0:.2f}s")

    # ── Stage 5: Quick detector test on face crops ──
    print("\n── Stage 5: Detector smoke test ──")

    # Frequency analysis on first few face crops
    freq = FrequencyAnalyzer()
    face_crops = []
    for fr in face_results:
        for face in fr.faces:
            face_crops.append(face.crop)
    face_crops = face_crops[:10]  # first 10

    if face_crops:
        freq_scores = [freq.analyze(c).score for c in face_crops]
        print(f"  [DCT] {len(freq_scores)} faces analyzed")
        print(f"        Scores: min={min(freq_scores):.3f}, max={max(freq_scores):.3f}, "
              f"mean={np.mean(freq_scores):.3f}")
    else:
        print("  [DCT] No face crops to analyze")

    # Temporal analysis on adjacent transition frame pairs
    temp = TemporalAnalyzer()
    transition_frames = [f for f in frames if f.selection_reason == "transition"]
    if len(transition_frames) >= 2:
        pairs_tested = 0
        temp_scores = []
        for i in range(len(transition_frames) - 1):
            fa, fb = transition_frames[i], transition_frames[i + 1]
            if abs(fa.frame_idx - fb.frame_idx) <= 5:  # only adjacent pairs
                result = temp.analyze_pair(fa.image, fb.image)
                temp_scores.append(result.score)
                pairs_tested += 1
        if temp_scores:
            print(f"  [TEMPORAL] {pairs_tested} pairs analyzed")
            print(f"             Scores: min={min(temp_scores):.3f}, max={max(temp_scores):.3f}, "
                  f"mean={np.mean(temp_scores):.3f}")
    else:
        print("  [TEMPORAL] Not enough transition frames for pair analysis")

    # ── Save debug output ──
    print("\n── Saving debug output ──")
    debug_dir = Path("data/reports/phase2_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save selected frames with face bounding boxes drawn
    for i, (sf, fr) in enumerate(zip(frames[:8], face_results[:8])):
        vis = sf.image.copy()
        for face in fr.faces:
            b = face.bbox
            cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 255, 0), 2)
            label = f"ID:{face.face_id} conf:{face.confidence:.2f}"
            cv2.putText(vis, label, (b.x, b.y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        out_path = debug_dir / f"frame_{i:03d}_idx{sf.frame_idx}.jpg"
        cv2.imwrite(str(out_path), vis)

    # Save face crops
    crop_dir = debug_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    crop_count = 0
    for fr in face_results:
        for face in fr.faces:
            out = crop_dir / f"face_id{face.face_id}_frame{face.frame_idx}.jpg"
            cv2.imwrite(str(out), face.crop)
            crop_count += 1
            if crop_count >= 20:
                break
        if crop_count >= 20:
            break

    # Save frequency spectrum for first face
    if face_crops:
        spectrum = freq.generate_spectrum_image(face_crops[0])
        cv2.imwrite(str(debug_dir / "dct_spectrum.jpg"), spectrum)

    print(f"  [OK] Debug images saved to {debug_dir}/")
    print(f"       - {min(8, len(frames))} annotated frames")
    print(f"       - {crop_count} face crops")
    print("       - 1 DCT spectrum visualization")

    # ── Summary ──
    print("\n" + "=" * 65)
    print("Phase 2 Integration Summary")
    print("=" * 65)
    print(f"  Video:          {meta.filename} ({meta.duration_sec:.1f}s)")
    print(f"  Scenes:         {scene_analysis.num_scenes}")
    print(f"  Frames sampled: {len(frames)} (budget: 30)")
    print(f"  Faces detected: {total_detections} across {frames_with_faces} frames")
    print(f"  Unique faces:   {len(unique_ids)}")
    print(f"  DCT scores:     {f'mean={np.mean(freq_scores):.3f}' if face_crops else 'N/A'}")
    print(f"  Debug output:   {debug_dir}/")
    print()

    if frames_with_faces == 0:
        print("  ⚠️  No faces detected! This could mean:")
        print("     - The video doesn't contain faces")
        print("     - Faces are too small (< 40px)")
        print("     - AI-generated faces aren't detected by MTCNN")
        print("     Try lowering min_confidence to 0.80 or min_face_size to 20")
    else:
        print("  ✅ Phase 2 complete. Ready for Phase 3 (ensemble + GradCAM).")
        print()
        print("  Next: Open data/reports/phase2_debug/ to inspect the output visually.")
        print("  Check that face crops look correct and bounding boxes are tight.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-find first video in sample_videos
        sample_dir = Path("data/sample_videos")
        videos = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.mov"))
        if videos:
            video_path = str(videos[0])
            print(f"Auto-detected: {video_path}")
        else:
            print("Usage: python scripts/test_phase2.py <video_path>")
            print("   or: place a video in data/sample_videos/")
            sys.exit(1)
    else:
        video_path = sys.argv[1]

    main(video_path)
