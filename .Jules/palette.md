## 2025-05-15 - Visual Input Controls for Node Schemas
**Learning:** Using visual hex color pickers and sliders for numeric inputs reduces cognitive load and makes the interface more intuitive compared to raw integer or float fields.
**Action:** Prefer `IO.Int.Input` with `display_mode=IO.NumberDisplay.color` for hex colors and `IO.Float.Input` with `display_mode=IO.NumberDisplay.slider` for normalized ranges (0.0-1.0).
