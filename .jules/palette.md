## 2025-07-21 - Visual controls for Node Inputs in ComfyUI
**Learning:** Node inputs representing colors or bounded ranges benefit immensely from specialized display modes rather than raw numeric text inputs. Displaying a visual color picker for hex-based integers or a slider for fractional value thresholds dramatically reduces user error and increases interface playfulness/usability.
**Action:** Always check schema definitions for integer/float inputs. If they represent colors, map them to `NumberDisplay.color`. If they represent normalized scales, map them to `NumberDisplay.slider`.
