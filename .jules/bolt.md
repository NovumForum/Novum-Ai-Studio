## 2026-03-06 - Separable 1D Gaussian Convolutions for Image Blur
**Learning:** 2D Gaussian filters ($K \times K$) are mathematically separable into horizontal ($1 \times K$) and vertical ($K \times 1$) 1D filters. Performing two 1D depthwise convolutions reduces FLOPs from $O(K^2)$ to $O(K)$, yielding a ~10x speedup for typical blur radii ($r=15, K=31$) while maintaining identical numerical outputs within floating point precision.
**Action:** When working with 2D spatial smoothing or separable filters, always prefer 1D horizontal and vertical passes over full 2D tensor operations.
