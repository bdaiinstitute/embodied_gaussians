# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
from isegm_gui import run_interactive_segmentor, load_model


class QuickSegmentor:

    def __init__(self, device="cuda"):
        self.device = device
        self.model = load_model('coco_lvis_h18_itermask.pth', device=self.device)

    def segment_with_gui(self, image: np.ndarray) -> np.ndarray | None:
        """Blocks and returns the mask. Mask is None if no points are selected.
        Mask is of type bool where True is foreground and False is background. 
        """

        assert image.ndim == 3, "Image must be 3D"
        assert image.shape[-1] == 3, "Image must be RGB"
        assert image.dtype == np.uint8, "Image must be uint8"

        gui = run_interactive_segmentor(self.model, device=self.device)
        gui.update_image(image)
        gui.mainloop()
        mask = gui.get_mask()
        try:
            gui.master.destroy()
        except:
            pass
        return mask
