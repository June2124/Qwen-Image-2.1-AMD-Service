import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from comfy_manager import ComfyProcessManager, ManagerConfig


class ManagerImageTests(unittest.TestCase):
    def manager(self, root):
        root = Path(root)
        return ComfyProcessManager(
            ManagerConfig(
                comfy_root=root / "comfy",
                input_dir=root / "input",
                output_dir=root / "output",
                user_dir=root / "user",
                frontend_root=root / "frontend",
                log_path=root / "comfy.log",
                host="127.0.0.1",
                port=8188,
                startup_timeout=1,
                inference_timeout=1,
                idle_unload_seconds=0,
                reserve_vram_gb=1.0,
                cache_device="auto",
                cache_dtype="default",
                reference_max_side=64,
            )
        )

    def test_canvas_is_edge_padded_without_resizing(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            source = Image.new("RGB", (65, 33), (10, 20, 30))
            source.putpixel((64, 32), (90, 100, 110))
            prepared = manager._prepare_input_image(source, is_canvas=True)
            self.assertEqual(prepared.size, (96, 64))
            self.assertEqual(prepared.getpixel((64, 32)), (90, 100, 110))
            self.assertEqual(prepared.getpixel((95, 63)), (90, 100, 110))

    def test_reference_is_downscaled_independently(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            source = Image.new("RGB", (160, 80), (10, 20, 30))
            prepared = manager._prepare_input_image(source, is_canvas=False)
            self.assertEqual(prepared.size, (64, 32))

    def test_generated_padding_is_cropped_to_original_canvas(self):
        source = Image.new("RGBA", (96, 64), (10, 20, 30, 255))
        encoded = io.BytesIO()
        source.save(encoded, format="PNG")
        raw = ComfyProcessManager._crop_output(encoded.getvalue(), (65, 33))
        with Image.open(io.BytesIO(raw)) as cropped:
            self.assertEqual(cropped.size, (65, 33))


if __name__ == "__main__":
    unittest.main()
