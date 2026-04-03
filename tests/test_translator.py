"""Unit tests for the request/response translation logic."""

from prism.translator.models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicTextBlock,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIResponse,
    OpenAIUsage,
)
from prism.translator.request import translate_request
from prism.translator.response import translate_response


class TestRequestTranslation:
    def test_simple_user_message(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
        )
        oai = translate_request(req)
        assert oai.model == "test-model"
        assert len(oai.messages) == 1
        assert oai.messages[0].role == "user"
        assert oai.messages[0].content == "Hello"
        assert oai.max_tokens == 100
        assert oai.stream is False

    def test_system_prompt_becomes_system_message(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system="You are helpful.",
        )
        oai = translate_request(req)
        assert oai.messages[0].role == "system"
        assert oai.messages[0].content == "You are helpful."
        assert oai.messages[1].role == "user"

    def test_content_blocks_flattened_to_string(self):
        req = AnthropicRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicTextBlock(text="Part 1"),
                        AnthropicTextBlock(text="Part 2"),
                    ],
                ),
            ],
        )
        oai = translate_request(req)
        assert oai.messages[0].content == "Part 1\nPart 2"

    def test_system_content_blocks(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system=[
                AnthropicTextBlock(text="Rule 1"),
                AnthropicTextBlock(text="Rule 2"),
            ],
        )
        oai = translate_request(req)
        assert oai.messages[0].content == "Rule 1\nRule 2"

    def test_stop_sequences_mapped_to_stop(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            stop_sequences=["END", "STOP"],
        )
        oai = translate_request(req)
        assert oai.stop == ["END", "STOP"]

    def test_optional_params_passed_through(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )
        oai = translate_request(req)
        assert oai.temperature == 0.7
        assert oai.top_p == 0.9
        assert oai.stream is True

    def test_top_k_is_dropped(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            top_k=40,
        )
        oai = translate_request(req)
        assert not hasattr(oai, "top_k")


class TestResponseTranslation:
    def test_basic_response(self):
        oai = OpenAIResponse(
            id="chatcmpl-123",
            model="test-model",
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(role="assistant", content="Hello back!"),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        resp = translate_response(oai, model="test-model")
        assert resp.role == "assistant"
        assert resp.content[0].text == "Hello back!"
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.id.startswith("msg_")

    def test_length_finish_reason(self):
        oai = OpenAIResponse(
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(role="assistant", content="..."),
                    finish_reason="length",
                )
            ],
        )
        resp = translate_response(oai, model="m")
        assert resp.stop_reason == "max_tokens"

    def test_empty_choices(self):
        oai = OpenAIResponse(choices=[])
        resp = translate_response(oai, model="m")
        assert resp.content[0].text == ""
        assert resp.stop_reason == "end_turn"
