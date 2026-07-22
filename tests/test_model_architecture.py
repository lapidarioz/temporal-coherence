"""Regression tests for the recovered temporal GAN architecture."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPOSITORY_ROOT / "model.py"


def _function_definition(name: str) -> ast.FunctionDef:
    module = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} was not found in {MODEL_PATH}")


def _assigned_list(function: ast.FunctionDef, variable_name: str) -> ast.List:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == variable_name:
            if not isinstance(node.value, ast.List):
                raise AssertionError(f"{variable_name} is not a list")
            return node.value
    raise AssertionError(f"Assignment to {variable_name!r} was not found")


def _block_filters(blocks: ast.List, function_name: str) -> list[int]:
    filters: list[int] = []
    for block in blocks.elts:
        if not isinstance(block, ast.Call) or not isinstance(block.func, ast.Name):
            raise AssertionError(f"Unexpected block expression: {ast.dump(block)}")
        if block.func.id != function_name:
            raise AssertionError(
                f"Expected {function_name} call, found {block.func.id!r}"
            )
        if not block.args or not isinstance(block.args[0], ast.Constant):
            raise AssertionError("Block filter count must be a literal")
        filters.append(int(block.args[0].value))
    return filters


def _keyword_values(blocks: ast.List, keyword_name: str) -> list[object | None]:
    values: list[object | None] = []
    for block in blocks.elts:
        if not isinstance(block, ast.Call):
            raise AssertionError(f"Unexpected block expression: {ast.dump(block)}")
        matching_keywords = [
            keyword for keyword in block.keywords if keyword.arg == keyword_name
        ]
        if not matching_keywords:
            values.append(None)
            continue
        if len(matching_keywords) != 1 or not isinstance(
            matching_keywords[0].value, ast.Constant
        ):
            raise AssertionError(f"Unexpected {keyword_name} declaration")
        values.append(matching_keywords[0].value.value)
    return values


class GeneratorSourceRegressionTests(unittest.TestCase):
    """Tests that do not import TensorFlow or download external assets."""

    def test_recovered_encoder_and_decoder_blocks(self) -> None:
        generator = _function_definition("get_generator_model")
        down_stack = _assigned_list(generator, "down_stack")
        up_stack = _assigned_list(generator, "up_stack")

        self.assertEqual(
            _block_filters(down_stack, "downsample"),
            [64, 128, 256, 512, 512, 512, 512, 512],
        )
        self.assertEqual(
            _block_filters(up_stack, "upsample"),
            [512, 512, 512, 512, 256, 128, 64],
        )
        self.assertEqual(
            _keyword_values(down_stack, "apply_batchnorm"),
            [False, None, None, None, None, None, None, None],
        )
        self.assertEqual(
            _keyword_values(up_stack, "apply_dropout"),
            [True, True, True, None, None, None, None],
        )

    def test_recovered_stacks_contain_no_ellipsis(self) -> None:
        generator = _function_definition("get_generator_model")
        for variable_name in ("down_stack", "up_stack"):
            blocks = _assigned_list(generator, variable_name)
            ellipses = [
                node
                for node in ast.walk(blocks)
                if isinstance(node, ast.Constant) and node.value is Ellipsis
            ]
            self.assertEqual(ellipses, [], f"{variable_name} contains an ellipsis")


@unittest.skipUnless(
    importlib.util.find_spec("tensorflow") is not None,
    "TensorFlow is not installed in this environment",
)
class ModelConstructionTests(unittest.TestCase):
    """Keras construction tests, enabled when TensorFlow is available."""

    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("legacy_model", MODEL_PATH)
        if spec is None or spec.loader is None:
            raise AssertionError(f"Could not load {MODEL_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.model_module = module

    def test_generator_inputs_and_output_shape(self) -> None:
        generator = self.model_module.get_generator_model(256, 256, 3)

        self.assertEqual(
            [tensor.name.split(":", maxsplit=1)[0] for tensor in generator.inputs],
            ["neutral_frame", "previous_frame", "warped_frame", "distances"],
        )
        self.assertEqual(generator.output_shape, (None, 256, 256, 3))

    def test_temporal_discriminator_inputs_and_output_shape(self) -> None:
        discriminator = self.model_module.get_discriminator_model(256, 256, 3)

        self.assertEqual(
            [tensor.name.split(":", maxsplit=1)[0] for tensor in discriminator.inputs],
            ["previous_frame", "current_frame", "difference"],
        )
        self.assertEqual(
            sum(int(tensor.shape[-1]) for tensor in discriminator.inputs),
            9,
        )
        self.assertEqual(discriminator.output_shape, (None, 30, 30, 1))


if __name__ == "__main__":
    unittest.main()
