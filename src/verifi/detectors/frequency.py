"""DCT frequency analysis — pure signal processing, no ML."""
import cv2
import numpy as np
from scipy.fft import dctn

from verifi.detectors.base import DetectionResult


class FrequencyAnalyzer:
    """
    Detect GAN/diffusion fingerprints in the frequency domain.
    Generators suppress high-frequency energy and leave periodic
    patterns from upsampling layers.
    """

    def __init__(self, baseline_hf_ratio: float = 0.38):
        self.baseline_hf_ratio = baseline_hf_ratio

    def analyze(self, face_crop: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        dct = dctn(gray, norm="ortho")
        magnitude = np.abs(dct)

        h, w = magnitude.shape
        mid_h, mid_w = h // 4, w // 4
        lf_energy = magnitude[:mid_h, :mid_w].sum()
        hf_energy = magnitude[mid_h:, :].sum() + magnitude[:mid_h, mid_w:].sum()

        total = lf_energy + hf_energy
        hf_ratio = float(hf_energy / total) if total > 0 else 0.5

        suppression = max(0, self.baseline_hf_ratio - hf_ratio) / self.baseline_hf_ratio
        score = min(1.0, suppression * 2.5)

        return DetectionResult(
            score=score,
            metadata={
                "hf_ratio": hf_ratio,
                "hf_suppression_pct": round(
                    (self.baseline_hf_ratio - hf_ratio) / self.baseline_hf_ratio * 100, 1
                ),
                "lf_energy": float(lf_energy),
                "hf_energy": float(hf_energy),
            },
        )

    def generate_spectrum_image(self, face_crop: np.ndarray) -> np.ndarray:
        """Generate a visual DCT spectrum image for the forensic report."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        dct = dctn(gray, norm="ortho")
        magnitude = np.log1p(np.abs(dct))

        # Normalize to 0-255 for visualization
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
        return colored
