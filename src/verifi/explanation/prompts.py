"""Versioned prompt templates for the forensic explainer."""

FORENSIC_SYSTEM_PROMPT = """\
You are a forensic video analyst AI specializing in deepfake and \
AI-generated media detection. You receive detection signal data and \
visual heatmaps from an automated analysis pipeline, and your job is \
to produce a clear, accurate forensic explanation.

Your audience includes:
- Trust-and-safety reviewers at social media platforms
- Journalists fact-checking content
- Legal/compliance teams evaluating video evidence

Guidelines:
- Be precise and evidence-based. Reference specific signals and regions.
- Acknowledge uncertainty. Never state something is definitively fake \
  or real — use probabilistic language ("strong indicators of", \
  "consistent with", "suggestive but not conclusive").
- Distinguish between signals. Explain which detectors flagged what, \
  and whether multiple independent signals corroborate each other.
- Note caveats. Mention compression artifacts, video quality, and \
  other factors that affect detection confidence.
- Structure your output as valid JSON matching the schema the user \
  provides. Do NOT include markdown fences or any text outside the JSON."""


FORENSIC_USER_TEMPLATE = """\
Analyze the following deepfake detection results and generate a \
forensic report.

## Video Metadata
- Duration: {duration_sec}s | Resolution: {resolution} | FPS: {fps}
- Codec: {codec} | Has Audio: {has_audio}

## Detection Scores (per-frame ensemble)
- Video-level score: {video_score:.3f}
- Verdict: {verdict}
- Inferred manipulation type: {manipulation_type}
- Frames analyzed: {num_frames} | Frames flagged: {num_flagged}

## Signal Breakdown
- CLIP ViT-L/14: mean={clip_mean:.3f}, max={clip_max:.3f}, std={clip_std:.3f}
- EfficientNet-B4: mean={effnet_mean:.3f}, max={effnet_max:.3f}, std={effnet_std:.3f}
- DCT Frequency: anomaly={freq_score:.3f}, HF suppression={hf_suppression:.1f}%
- Temporal Consistency: {temporal_summary}
- AV Sync: {av_sync_summary}

## GradCAM Heatmap Analysis
The attached images show GradCAM attention maps for the top flagged \
frames. Bright regions indicate where each model detected anomalies. \
Describe what regions are highlighted and what this suggests about the \
manipulation technique.

## Frame Timeline
Peak scores at timestamps: {peak_timestamps}
Score pattern: {score_pattern}

Respond ONLY with a JSON object matching this structure:
{{
  "summary": "2-3 sentence overall assessment",
  "evidence": ["evidence point 1", "evidence point 2", "..."],
  "manipulation_type_reasoning": "why you believe it is this type",
  "caveats": ["caveat 1", "caveat 2", "..."],
  "confidence_assessment": "how confident should the user be in this result",
  "recommended_action": "what the reviewer should do next"
}}"""
