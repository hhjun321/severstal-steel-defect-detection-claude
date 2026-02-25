"""
Hint Image Generation Module for ControlNet

Two hint modes are supported:

  canny (v5, default):
    Canny edge detection applied to the defect mask boundary.
    This is the correct format for the pretrained ``sd-controlnet-canny``
    weights that are used as the starting point for fine-tuning.
    Produces white edge lines on a black background (3-channel, R==G==B).

  grayscale (v4, legacy):
    Three feature channels (R=defect mask, G=background edges,
    B=texture density) combined via weighted summation
    (R*0.5 + G*0.3 + B*0.2) and replicated across all 3 channels.
    This format is INCOMPATIBLE with sd-controlnet-canny pretrained
    weights, causing the model to ignore hints (guidance_scale ↑ →
    quality_score ↓ pattern observed in v4 Phase 2 results).
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from pathlib import Path


class HintImageGenerator:
    """
    Generates hint images for ControlNet training.

    Supports two modes:
      - ``canny``: Canny edges of the defect mask boundary (v5, compatible
        with ``sd-controlnet-canny`` pretrained weights).
      - ``grayscale``: Legacy multi-channel weighted grayscale (v4).
    """

    def __init__(self, enhance_linearity: bool = True,
                 enhance_background: bool = True):
        """
        Initialize hint image generator.

        Args:
            enhance_linearity: Whether to enhance linear defects in red channel
                (grayscale mode only)
            enhance_background: Whether to enhance background structures
                (grayscale mode only)
        """
        self.enhance_linearity = enhance_linearity
        self.enhance_background = enhance_background

    # ------------------------------------------------------------------
    # v5: Canny hint (correct format for sd-controlnet-canny)
    # ------------------------------------------------------------------

    def generate_canny_hint(
        self,
        roi_mask: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
        dilate_iterations: int = 1,
    ) -> np.ndarray:
        """
        Generate Canny edge hint from defect mask boundary (v5).

        The output is a 3-channel image (R==G==B) with white edges on a
        black background, which is exactly the format expected by the
        ``sd-controlnet-canny`` pretrained ControlNet.

        Args:
            roi_mask: Binary defect mask (H, W), values 0 or 1.
            low_threshold: Canny low hysteresis threshold (default 50).
            high_threshold: Canny high hysteresis threshold (default 150).
            dilate_iterations: Dilation passes to make edges visible
                (default 1; set to 0 to disable).

        Returns:
            3-channel hint image (H, W, 3) with uint8 values in [0, 255].
        """
        mask_uint8 = (roi_mask > 0).astype(np.uint8) * 255
        edges = cv2.Canny(mask_uint8, low_threshold, high_threshold)

        if dilate_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

        return np.stack([edges, edges, edges], axis=2)

    # ------------------------------------------------------------------
    # v4 / legacy: Grayscale hint (kept for backward compatibility)
    # ------------------------------------------------------------------

    def generate_red_channel(self, mask: np.ndarray,
                            defect_metrics: Dict) -> np.ndarray:
        """
        Generate RED channel: Defect mask with 4-indicator enhancement.

        The intensity is modulated by defect characteristics:
        - High linearity → sharper, more defined edges
        - High solidity → filled regions
        - Low solidity → edge-emphasized regions

        Args:
            mask: Binary defect mask (H, W)
            defect_metrics: Dictionary with linearity, solidity, extent, aspect_ratio

        Returns:
            Red channel (H, W) with values 0-255
        """
        h, w = mask.shape
        red_channel = np.zeros((h, w), dtype=np.uint8)

        if mask.sum() == 0:
            return red_channel

        # Base mask intensity
        base_intensity = 200

        # Get metrics
        linearity = defect_metrics.get('linearity', 0.5)
        solidity = defect_metrics.get('solidity', 0.5)

        if self.enhance_linearity and linearity > 0.7:
            # For high linearity (scratches), emphasize edges
            # Use morphological skeleton (Zhang-Suen algorithm via scikit-image)
            from skimage.morphology import skeletonize

            skeleton = skeletonize(mask > 0)

            # Dilate slightly to make visible
            kernel = np.ones((3, 3), np.uint8)
            skeleton_uint8 = (skeleton * 255).astype(np.uint8)
            enhanced_skeleton = cv2.dilate(skeleton_uint8, kernel, iterations=1)

            red_channel = np.where(enhanced_skeleton > 0, 255, 0).astype(np.uint8)

        elif solidity > 0.8:
            # High solidity (compact blobs) - use filled mask
            red_channel = np.where(mask > 0, base_intensity, 0).astype(np.uint8)

        else:
            # General case - edge-emphasized mask
            # Compute edges
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)

            # Dilate edges
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # Combine with mask
            red_channel = np.where(mask > 0, base_intensity // 2, 0).astype(np.uint8)
            red_channel = np.where(dilated_edges > 0, 255, red_channel).astype(np.uint8)

        return red_channel

    def generate_green_channel(self, image: np.ndarray,
                               background_type: str,
                               stability_score: float) -> np.ndarray:
        """
        Generate GREEN channel: Background structure lines (edges from patterns).

        Emphasizes directional patterns:
        - vertical_stripe: Vertical edges
        - horizontal_stripe: Horizontal edges
        - complex_pattern: All edges
        - smooth/textured: Minimal edges

        Args:
            image: Original ROI image (H, W, 3) or (H, W)
            background_type: Background classification
            stability_score: Background stability (0-1)

        Returns:
            Green channel (H, W) with values 0-255
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        green_channel = np.zeros((h, w), dtype=np.uint8)

        if not self.enhance_background:
            return green_channel

        # Compute gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        if background_type == 'vertical_stripe':
            # Emphasize vertical edges (horizontal gradients)
            edge_map = np.abs(sobel_x)

        elif background_type == 'horizontal_stripe':
            # Emphasize horizontal edges (vertical gradients)
            edge_map = np.abs(sobel_y)

        elif background_type == 'complex_pattern':
            # Use all edges
            edge_map = np.sqrt(sobel_x**2 + sobel_y**2)

        else:  # smooth or textured
            # Minimal edge information
            edge_map = np.sqrt(sobel_x**2 + sobel_y**2) * 0.3

        # Normalize to 0-255
        edge_map_norm = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)

        # Modulate by stability score
        # Higher stability → clearer structure lines
        intensity_factor = 0.5 + 0.5 * stability_score
        green_channel = (edge_map_norm * intensity_factor).astype(np.uint8)

        return green_channel

    def generate_blue_channel(self, image: np.ndarray,
                             background_type: str) -> np.ndarray:
        """
        Generate BLUE channel: Background fine texture or noise density.

        Captures high-frequency texture information:
        - smooth: Low values
        - textured/complex: High-frequency components

        Args:
            image: Original ROI image (H, W, 3) or (H, W)
            background_type: Background classification

        Returns:
            Blue channel (H, W) with values 0-255
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        if background_type == 'smooth':
            # Very low texture
            blue_channel = np.ones((h, w), dtype=np.uint8) * 20

        else:
            # Compute local variance (texture measure)
            # Use a sliding window approach
            kernel_size = 7
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

            # Compute local mean
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)

            # Compute local variance
            local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            local_var = local_sq_mean - local_mean**2
            local_var = np.maximum(local_var, 0)  # Ensure non-negative

            # Normalize to 0-255
            blue_channel = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Adjust intensity based on background type
            if background_type in ['textured', 'complex_pattern']:
                # Higher texture emphasis
                blue_channel = np.clip(blue_channel * 1.2, 0, 255).astype(np.uint8)

        return blue_channel

    def generate_hint_image(self, roi_image: np.ndarray,
                           roi_mask: np.ndarray,
                           defect_metrics: Dict,
                           background_type: str,
                           stability_score: float,
                           hint_mode: str = 'grayscale') -> np.ndarray:
        """
        Generate hint image for ControlNet (dispatcher for v4/v5 modes).

        Args:
            roi_image: Original ROI image (H, W, 3).  Used in ``grayscale`` mode only.
            roi_mask: Binary defect mask (H, W).
            defect_metrics: Defect characterization metrics.  Used in ``grayscale`` mode only.
            background_type: Background classification.  Used in ``grayscale`` mode only.
            stability_score: Background stability score.  Used in ``grayscale`` mode only.
            hint_mode: ``'canny'`` (v5, recommended) or ``'grayscale'`` (v4, legacy).

        Returns:
            3-channel hint image (H, W, 3).
        """
        if hint_mode == 'canny':
            return self.generate_canny_hint(roi_mask)

        # ---- grayscale (v4 legacy) ----
        red = self.generate_red_channel(roi_mask, defect_metrics)
        green = self.generate_green_channel(roi_image, background_type, stability_score)
        blue = self.generate_blue_channel(roi_image, background_type)

        # Weighted grayscale conversion to eliminate RGB color signals.
        w_r, w_g, w_b = 0.5, 0.3, 0.2
        gray = (w_r * red.astype(np.float32)
                + w_g * green.astype(np.float32)
                + w_b * blue.astype(np.float32))
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Replicate the single grayscale channel across all 3 channels
        hint_image = np.stack([gray, gray, gray], axis=2)

        return hint_image

    def save_hint_image(self, hint_image: np.ndarray, output_path: Path):
        """
        Save hint image to file.

        Args:
            hint_image: 3-channel hint image (H, W, 3)
            output_path: Output file path
        """
        # Convert RGB to BGR for OpenCV
        hint_bgr = cv2.cvtColor(hint_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), hint_bgr)

    def visualize_channels(self, hint_image: np.ndarray) -> np.ndarray:
        """
        Create visualization showing each channel separately.

        Args:
            hint_image: 3-channel hint image (H, W, 3)

        Returns:
            Visualization image showing all channels side by side
        """
        red = hint_image[:, :, 0]
        green = hint_image[:, :, 1]
        blue = hint_image[:, :, 2]

        # Convert single channels to RGB for visualization
        red_vis = np.stack([red, np.zeros_like(red), np.zeros_like(red)], axis=2)
        green_vis = np.stack([np.zeros_like(green), green, np.zeros_like(green)], axis=2)
        blue_vis = np.stack([np.zeros_like(blue), np.zeros_like(blue), blue], axis=2)

        # Concatenate horizontally
        visualization = np.concatenate([hint_image, red_vis, green_vis, blue_vis], axis=1)

        return visualization
