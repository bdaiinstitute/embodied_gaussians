# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import cv2

from sam2.sam2_image_predictor import SAM2ImagePredictor


class SamSegmentor:

    def __init__(self, device="cuda"):
        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-large", device=device
        )

    def fill_holes(self, mask: np.ndarray):
        mask = mask.astype(np.uint8)
        inverted_mask = cv2.bitwise_not(mask)
        h, w = inverted_mask.shape[:2]
        flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(inverted_mask, flood_fill_mask, (0, 0), 255)
        filled_region = flood_fill_mask[1:-1, 1:-1]
        final_mask = np.logical_not(filled_region).astype(bool)
        return final_mask

    def segment_with_gui(self, image: np.ndarray, fill_holes: bool = True) -> np.ndarray | None:
        """Blocks and returns the mask. Mask is None if no points are selected.
        Mask is of type bool where True is foreground and False is background. 
        """

        assert image.ndim == 3, "Image must be 3D"
        assert image.shape[-1] == 3, "Image must be RGB"
        assert image.dtype == np.uint8, "Image must be uint8"

        self.predictor.set_image(image)
        mask = np.zeros(image.shape[:2], dtype=bool)

        # VIS STUFF
        foreground_points = []
        background_points = []
        image_vis = image.copy()
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
        alpha = 0.5

        def click_event(event, x, y, _, _2):
            nonlocal mask
            if event == cv2.EVENT_RBUTTONDOWN:
                background_points.append([x, y])
            elif event == cv2.EVENT_LBUTTONDOWN:
                foreground_points.append([x, y])
            elif event == cv2.EVENT_MBUTTONDOWN:
                # clear
                foreground_points.clear()
                background_points.clear()
                mask = np.zeros(image.shape[:2], dtype=bool)
            else:
                return


            all_points = foreground_points + background_points
            labels = [1] * len(foreground_points) + [0] * len(background_points)
            if len(all_points) > 0:
                masks, _, _ = self.predictor.predict(
                    point_coords=np.asarray(all_points),
                    point_labels=np.asarray(labels),
                    multimask_output=False,
                )
                mask = masks[0].astype(bool)
                if fill_holes:
                    mask = self.fill_holes(mask)

            overlay = image.copy()
            overlay[mask] = (0, 191, 255)
            blended = cv2.addWeighted(overlay, alpha, image_vis, 1 - alpha, 0)
            cv2.imshow("Quick Segmentor", blended)

        cv2.namedWindow("Quick Segmentor", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Quick Segmentor", 1280, 720)
        cv2.setMouseCallback("Quick Segmentor", click_event)
        cv2.imshow("Quick Segmentor", image_vis)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        if len(foreground_points) == 0:
            return None

        return mask
