## 2025-05-15 - Visual affordances for mask nodes
**Learning:** Prefer using visual hex color pickers and sliders for bounded numeric inputs to reduce cognitive load and improve precision. Even in backend-heavy environments like ComfyUI, providing UI hints through node schema metadata is a powerful micro-UX win.
**Action:** Always check if integer inputs represent colors (use `NumberDisplay.color`) or if float inputs are bounded (use `NumberDisplay.slider`).
