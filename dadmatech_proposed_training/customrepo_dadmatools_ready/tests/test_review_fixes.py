import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train


class FakeTokenizer:
    def __init__(self, template):
        self.chat_template = template
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return [1, 2, 3]


QWEN_LIKE = "Chat template with enable_thinking switch."
PLAIN_TEMPLATE = "Chat template without the switch."


class TemplateSupportTests(unittest.TestCase):
    def test_support_detection(self):
        self.assertTrue(train._template_supports_enable_thinking(FakeTokenizer(QWEN_LIKE)))
        self.assertFalse(train._template_supports_enable_thinking(FakeTokenizer(PLAIN_TEMPLATE)))
        self.assertFalse(train._template_supports_enable_thinking(FakeTokenizer(None)))

    def test_apply_template_passes_flag_when_supported(self):
        tok = FakeTokenizer(QWEN_LIKE)
        train._apply_template(tok, [{"role": "user", "content": "hi"}], None,
                              add_generation_prompt=True, enable_thinking=False)
        self.assertIs(tok.calls[0]["enable_thinking"], False)

    def test_apply_template_omits_flag_when_unsupported(self):
        tok = FakeTokenizer(PLAIN_TEMPLATE)
        train._apply_template(tok, [{"role": "user", "content": "hi"}], None,
                              add_generation_prompt=True, enable_thinking=False)
        self.assertNotIn("enable_thinking", tok.calls[0])

    def test_apply_template_raises_for_thinking_on_unsupported(self):
        tok = FakeTokenizer(PLAIN_TEMPLATE)
        with self.assertRaises(RuntimeError):
            train._apply_template(tok, [{"role": "user", "content": "hi"}], None,
                                  add_generation_prompt=True, enable_thinking=True)

    def test_tools_forwarded(self):
        tok = FakeTokenizer(QWEN_LIKE)
        tools = [{"type": "function", "function": {"name": "x"}}]
        train._apply_template(tok, [{"role": "user", "content": "hi"}], tools,
                              add_generation_prompt=True, enable_thinking=True)
        self.assertEqual(tok.calls[0]["tools"], tools)


class ResolveThinkingModesTests(unittest.TestCase):
    def test_auto_without_reasoning(self):
        self.assertEqual(train.resolve_thinking_modes("auto", False), [False])

    def test_auto_with_reasoning(self):
        self.assertEqual(train.resolve_thinking_modes("auto", True), [True])

    def test_forced_modes(self):
        self.assertEqual(train.resolve_thinking_modes("true", False), [True])
        self.assertEqual(train.resolve_thinking_modes("FALSE", False), [False])
        self.assertEqual(train.resolve_thinking_modes("both", False), [True, False])
        self.assertEqual(train.resolve_thinking_modes("", True), [True])

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            train.resolve_thinking_modes("sometimes", False)


class DatasetHasReasoningTests(unittest.TestCase):
    def test_detects_reasoning_field(self):
        rows = [{"messages": [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "a",
                               "reasoning_content": "trace"}]}]
        self.assertTrue(train.dataset_has_reasoning(iter(rows)))

    def test_detects_think_tag(self):
        rows = [{"messages": [{"role": "user", "content": "q"},
                              {"role": "assistant",
                               "content": "<think>secret</think>final"}]}]
        self.assertTrue(train.dataset_has_reasoning(iter(rows)))

    def test_no_reasoning(self):
        rows = [{"messages": [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "a"}]}]
        self.assertFalse(train.dataset_has_reasoning(iter(rows)))


class ConfigDefaultsTests(unittest.TestCase):
    def test_doc_defaults(self):
        self.assertEqual(train.CONFIG["epochs"], 1.0)
        self.assertEqual(train.CONFIG["batch_size"], 2)
        self.assertEqual(train.CONFIG["grad_accum"], 4)
        self.assertEqual(train.CONFIG["optim"], "adamw_8bit")
        self.assertEqual(train.CONFIG["learning_rate"], 1e-4)
        self.assertEqual(train.CONFIG["lora_rank"], 32)
        self.assertEqual(train.CONFIG["lora_alpha"], 16)
        self.assertEqual(train.CONFIG["lora_dropout"], 0.05)
        self.assertEqual(train.CONFIG["warmup_epochs"], 0.1)
        self.assertEqual(train.CONFIG["thinking_mode"], "auto")
        self.assertIsNone(train.CONFIG["dataset_dir"])  # registry download by default
        self.assertNotIn("strict_thinking_template", train.CONFIG)
        self.assertNotIn("thinking_modes", train.CONFIG)


class SftConfigTests(unittest.TestCase):
    def test_fields(self):
        try:
            from trl import SFTConfig
        except ImportError:
            self.skipTest("trl not installed")
        cfg = dict(train.CONFIG)
        run_dir = Path("test_sft_config_tmp")
        args = train._sft_config(cfg, run_dir)
        self.assertEqual(args.per_device_train_batch_size, 2)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertEqual(args.optim.value if hasattr(args.optim, "value") else str(args.optim),
                         "adamw_8bit")
        self.assertTrue(args.save_only_model)
        self.assertEqual(args.save_total_limit, 1)
        self.assertAlmostEqual(args.warmup_ratio, 0.1)


if __name__ == "__main__":
    unittest.main()
