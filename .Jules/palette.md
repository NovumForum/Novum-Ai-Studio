# Palette's Journal - Critical UX/Accessibility Learnings

## 2025-05-15 - Micro-UX Improvements for Mask Nodes
**Learning:** Visualizing numeric inputs as sliders or color pickers significantly reduces cognitive load for users by providing immediate context on the range and nature of the input.
**Action:** Always check if an `INT` or `FLOAT` input represents a color or a normalized range (0-1) and apply the appropriate `NumberDisplay` mode.
