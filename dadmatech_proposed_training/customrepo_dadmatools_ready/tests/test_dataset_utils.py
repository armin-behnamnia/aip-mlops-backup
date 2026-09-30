import unittest

from dataset_utils import final_inference_case, normalize_example, without_reasoning


class DatasetUtilsTests(unittest.TestCase):
    def test_openai_messages_with_tool_calls(self):
        row = {
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function",
                         "function": {"name": "weather", "arguments": {"city": "Tehran"}}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "27 C"},
                {"role": "assistant", "content": "It is 27 C."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        messages, tools = normalize_example(row, None)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "weather")

    def test_functions_field_and_json_string_tools(self):
        row = {
            "messages": [{"role": "user", "content": "Run it"}],
            "functions": '[{"name": "run", "description": "Run",'
                         '"parameters": {"type": "object", "properties": {}}}]',
        }
        _, tools = normalize_example(row, "System")
        self.assertEqual(tools[0]["function"]["name"], "run")

    def test_openai_function_call_is_converted(self):
        row = {
            "messages": [
                {"role": "user", "content": "Run it"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "run", "arguments": "{}"},
                },
            ]
        }
        messages, _ = normalize_example(row)
        self.assertEqual(messages[1]["content"], "")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_sharegpt_conversations_rejected(self):
        row = {"conversations": [{"from": "human", "value": "Hi"}]}
        with self.assertRaises(Exception):
            normalize_example(row)

    def test_function_call_role_message(self):
        # Defensive: some ingestion outputs keep tool-call turns under this
        # role with the call JSON in content (single or parallel calls).
        row = {
            "messages": [
                {"role": "user", "content": "Compare these."},
                {"role": "function_call",
                 "content": '[{"name": "lookup", "arguments": {"id": 1}},'
                            '{"name": "lookup", "arguments": {"id": 2}}]'},
                {"role": "tool", "content": "results"},
                {"role": "assistant", "content": "The comparison."},
            ]
        }
        messages, _ = normalize_example(row, None)
        self.assertEqual(
            [m["role"] for m in messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(len(messages[1]["tool_calls"]), 2)
        self.assertEqual(messages[1]["tool_calls"][1]["function"]["name"], "lookup")

    def test_tool_calls_json_string_element_flattened(self):
        # Defensive: a tool_calls element may be a JSON string holding either
        # one call object or an array of parallel calls.
        row = {
            "messages": [
                {"role": "user", "content": "Run"},
                {"role": "assistant", "content": "",
                 "tool_calls": ['{"name": "run", "arguments": {}}']},
                {"role": "tool", "content": "done"},
                {"role": "assistant", "content": "Finished."},
            ]
        }
        messages, _ = normalize_example(row, None)
        self.assertEqual(len(messages[1]["tool_calls"]), 1)
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "run")

    def test_alpaca_fallback(self):
        messages, _ = normalize_example(
            {"instruction": "hi", "input": "ctx", "output": "o"}, "sys"
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi\n\nctx"},
                {"role": "assistant", "content": "o"},
            ],
        )

    def test_nonthinking_removes_reasoning_only(self):
        message = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "private trace",
            "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": {}}}],
        }
        result = without_reasoning(message)
        self.assertNotIn("reasoning_content", result)
        self.assertIn("tool_calls", result)

    def test_final_assistant_is_test_target(self):
        history, expected, _ = final_inference_case(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(expected["content"], "Hello")


if __name__ == "__main__":
    unittest.main()
