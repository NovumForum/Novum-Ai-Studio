## 2025-05-15 - Visual Hinting for Specialized Inputs
**Learning:** In node-based UIs, providing visual hints like color pickers for hex values and sliders for normalized floats (0-1) significantly reduces cognitive load compared to raw numeric inputs. The `IO.Int.Input` and `IO.Float.Input` in ComfyUI support a `display_mode` parameter that can be used to trigger these UI components.

**Action:** Use `display_mode=IO.NumberDisplay.color` for integer inputs representing hex colors and `display_mode=IO.NumberDisplay.slider` for float/int inputs with defined ranges that represent thresholds or strengths.
