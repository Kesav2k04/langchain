from typing import Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI


class SomeStructuredResp(BaseModel):
    response: str = Field("default reasoning")


def test_reasoning_effort_does_not_force_responses_api(mocker: Any) -> None:
    llm = ChatOpenAI(model="o1-preview", reasoning_effort="low", api_key="fake")

    # We want to intercept the final request payload to make sure it doesn't use
    # Responses API. Since we can't easily intercept the payload without sending
    # the request and mocking the client, we can just mock
    # `self.client.chat.completions.create` to return a dummy response.

    structured_llm = llm.with_structured_output(SomeStructuredResp)

    # Mock the _use_responses_api to verify if it would be True or False.
    # We can just assert that `_use_responses_api` is False for our payload.

    payload = {
        "model": "o1-preview",
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "hi"}],
    }

    assert not llm._use_responses_api(payload), (
        "reasoning_effort should not force Responses API"
    )

    # We can also mock `_get_request_payload` to check that the payload still
    # has `response_format` instead of `text_format`.
    original_get_payload = llm._get_request_payload
    payload_used = None

    def mock_get_request_payload(*args: Any, **kwargs: Any) -> dict:
        nonlocal payload_used
        payload_used = original_get_payload(*args, **kwargs)
        return payload_used

    mocker.patch.object(
        llm, "_get_request_payload", side_effect=mock_get_request_payload
    )

    # Mock the synchronous OpenAI client to avoid network call
    class MockCompletions:
        def parse(self, **kwargs: Any) -> Any:
            pass

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    mocker.patch.object(llm, "root_client", MockClient())
    mocker.patch.object(llm, "client", MockClient())

    # mock _generate to just run _get_request_payload and return a dummy
    # Actually, if we mock the root_client, it will fail inside `parse`.
    # But that's fine, we just want to catch the payload before it fails.

    try:
        structured_llm.invoke("hello")
    except Exception:  # noqa: S110
        # Expected since mock does not return real response
        pass

    assert payload_used is not None
    assert "response_format" in payload_used
    assert "text_format" not in payload_used
