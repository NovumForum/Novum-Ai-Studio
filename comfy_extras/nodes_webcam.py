import nodes
import folder_paths

MAX_RESOLUTION = nodes.MAX_RESOLUTION


class WebcamCapture(nodes.LoadImage):
    DESCRIPTION = "Captures image snapshots directly from a connected webcam device."
    SEARCH_ALIASES = ["camera input", "live capture", "camera feed", "snapshot"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("WEBCAM", {"tooltip": "The webcam snapshot source or captured frame path."}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Target output frame width in pixels. Set to 0 to preserve source camera resolution."}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Target output frame height in pixels. Set to 0 to preserve source camera resolution."}),
                "capture_on_queue": ("BOOLEAN", {"default": True, "tooltip": "When enabled, captures a fresh image snapshot automatically whenever the workflow prompt is queued."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_capture"

    CATEGORY = "image"

    def load_capture(self, image, **kwargs):
        return super().load_image(folder_paths.get_annotated_filepath(image))

    @classmethod
    def IS_CHANGED(cls, image, width, height, capture_on_queue):
        return super().IS_CHANGED(image)


NODE_CLASS_MAPPINGS = {
    "WebcamCapture": WebcamCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebcamCapture": "Webcam Capture",
}
