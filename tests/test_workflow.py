import unittest

from comfy_workflow import (
    SIZE_PRESETS,
    WorkflowValidationError,
    build_workflow,
    canvas_workflow_size,
    parse_size,
    reference_resolution,
)


class WorkflowTests(unittest.TestCase):
    def test_text_to_image_has_no_load_image_nodes(self):
        workflow = build_workflow(
            mode="text-to-image",
            prompt="a quiet room",
            negative_prompt="",
            size="640x640",
            steps=25,
            guidance_scale=1.0,
            seed=7,
        )
        self.assertFalse(
            any(node["class_type"] == "LoadImage" for node in workflow.values())
        )
        self.assertEqual(workflow["6"]["inputs"]["width"], 640)
        self.assertEqual(workflow["7"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(workflow["7"]["inputs"]["latent_image"], ["6", 0])
        self.assertEqual(workflow["2"]["inputs"]["dtype"], "default")

    def test_canvas_is_image_one_and_references_follow(self):
        workflow = build_workflow(
            mode="image-to-image",
            prompt="edit image 1 using image 2",
            negative_prompt="text, watermark",
            size="640x352",
            steps=40,
            guidance_scale=2.0,
            seed=11,
            image_names=["canvas.png", "reference.png"],
        )
        inputs = workflow["5"]["inputs"]
        self.assertEqual(inputs["images.image_1"], ["21", 0])
        self.assertEqual(inputs["images.image_2"], ["22", 0])
        self.assertEqual(workflow["21"]["inputs"]["image"], "canvas.png")
        self.assertEqual(workflow["22"]["inputs"]["image"], "reference.png")
        self.assertEqual(workflow["6"]["inputs"]["height"], 352)
        self.assertEqual(inputs["resolution"], 0)
        self.assertEqual(workflow["7"]["inputs"]["latent_image"], ["5", 2])

    def test_rejects_more_than_ten_images(self):
        with self.assertRaises(WorkflowValidationError):
            build_workflow(
                mode="image-to-image",
                prompt="edit",
                negative_prompt="",
                size="640x640",
                steps=25,
                guidance_scale=1.0,
                seed=1,
                image_names=["{}.png".format(index) for index in range(11)],
            )

    def test_all_presets_are_aligned(self):
        for width, height in SIZE_PRESETS.values():
            self.assertEqual(width % 32, 0)
            self.assertEqual(height % 32, 0)
            self.assertEqual(reference_resolution(width, height) % 32, 0)

    def test_canvas_size_is_padded_up_without_scaling(self):
        self.assertEqual(canvas_workflow_size(1920, 1072), (1920, 1088))
        self.assertEqual(canvas_workflow_size(640, 352), (640, 352))

    def test_dynamic_aligned_workflow_sizes_are_supported(self):
        self.assertEqual(parse_size("1920x1088"), (1920, 1088))
        with self.assertRaises(WorkflowValidationError):
            parse_size("1920x1072")


if __name__ == "__main__":
    unittest.main()
