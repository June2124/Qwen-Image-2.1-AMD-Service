import unittest

from comfy_workflow import TURBO_GUIDANCE_SCALE, TURBO_STEPS
from qwen_service import _payload


class ServicePayloadTests(unittest.TestCase):
    def test_legacy_model_and_sampling_parameters_are_normalized(self):
        values = _payload(
            "text-to-image",
            {
                "model": "qwen-image-2.1",
                "input": {"prompt": "a quiet room"},
                "parameters": {"steps": 50, "guidance_scale": 4.0, "seed": 42},
            },
        )

        self.assertEqual(values["steps"], TURBO_STEPS)
        self.assertEqual(values["guidance_scale"], TURBO_GUIDANCE_SCALE)
        self.assertEqual(values["requested_steps"], 50)
        self.assertEqual(values["requested_guidance_scale"], 4.0)
        self.assertEqual(values["seed"], 42)


if __name__ == "__main__":
    unittest.main()
