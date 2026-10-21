import ast


class TestInpaintAndNoiseOptimizationAST:

    def test_vae_encode_for_inpaint_vectorized(self):
        """Verify VAEEncodeForInpaint.encode contains no unrolled channel loops."""
        with open("nodes.py", "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        vae_class = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "VAEEncodeForInpaint":
                vae_class = node
                break

        assert vae_class is not None, "VAEEncodeForInpaint class not found in nodes.py"

        encode_method = None
        for node in vae_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "encode":
                encode_method = node
                break

        assert encode_method is not None, "encode method not found in VAEEncodeForInpaint"

        for_loops = [n for n in ast.walk(encode_method) if isinstance(n, ast.For)]
        assert len(for_loops) == 0, f"Expected 0 for loops in VAEEncodeForInpaint.encode, found {len(for_loops)}"

    def test_inpaint_model_conditioning_vectorized(self):
        """Verify InpaintModelConditioning.encode contains no unrolled channel loops."""
        with open("nodes.py", "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        inpaint_class = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "InpaintModelConditioning":
                inpaint_class = node
                break

        assert inpaint_class is not None, "InpaintModelConditioning class not found in nodes.py"

        encode_method = None
        for node in inpaint_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "encode":
                encode_method = node
                break

        assert encode_method is not None, "encode method not found in InpaintModelConditioning"

        for_loops = [n for n in ast.walk(encode_method) if isinstance(n, ast.For)]
        # Should only have 1 loop over [positive, negative]
        assert len(for_loops) == 1, f"Expected 1 for loop in InpaintModelConditioning.encode, found {len(for_loops)}"

    def test_image_add_noise_device_direct(self):
        """Verify ImageAddNoise.execute allocates noise directly on image.device."""
        with open("comfy_extras/nodes_images.py", "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        noise_class = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ImageAddNoise":
                noise_class = node
                break

        assert noise_class is not None, "ImageAddNoise class not found in comfy_extras/nodes_images.py"

        execute_method = None
        for node in noise_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                execute_method = node
                break

        assert execute_method is not None, "execute method not found in ImageAddNoise"

        method_source = ast.unparse(execute_method)
        assert 'device="cpu"' not in method_source and "device='cpu'" not in method_source, "Expected no CPU allocations in ImageAddNoise.execute"
        assert "torch.Generator(device=image.device)" in method_source, "Expected local device generator in ImageAddNoise.execute"
