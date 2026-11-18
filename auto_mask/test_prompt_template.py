"""Guards for the shared Qwen -> SAM3 prompt template (no model services needed)."""

import re

from vllm_space.sam3_prompt import PROMPT_TEMPLATE, PROMPT_VERSION


def test_template_formats_with_only_task_description():
    rendered = PROMPT_TEMPLATE.format(task_description="Lift the bottle.")
    assert "Lift the bottle." in rendered
    assert "{" not in rendered.replace("{task_description}", "")


def test_template_requires_both_robot_prompts():
    assert "Always start with exactly: robot, robot arm" in PROMPT_TEMPLATE


def test_example_outputs_never_use_words_sam3_misses():
    # Words measured to be (near) undetectable by SAM3 on RoboTwin frames.
    missed = {"stamp", "rubber stamp", "seal", "fan", "rack", "mug rack", "hamburger",
              "switch", "scanner", "tea box", "pad", "mat", "skillet", "bin"}
    outputs = re.findall(r"^Output: (.+)$", PROMPT_TEMPLATE, flags=re.MULTILINE)
    assert outputs, "template should contain worked examples"
    for line in outputs:
        phrases = [p.strip() for p in line.split(",")]
        assert phrases[:2] == ["robot", "robot arm"]
        assert not missed.intersection(phrases), line


def test_version_tag():
    assert PROMPT_VERSION.startswith("robotwin-sam3-v3")
