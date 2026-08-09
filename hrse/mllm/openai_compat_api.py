import base64
import mimetypes
import os
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _image_content(image_path: str) -> dict:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = _encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def _response_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                chunks.append(item.text)
        return "\n".join(chunks)
    return str(content or "")


class QnAIGCVision:
    """OpenAI-compatible vision client for the configured QnAIGC relay."""

    def __init__(self):
        api_key = os.environ.get("QNAIGC_API_KEY")
        base_url = os.environ.get("QNAIGC_BASE_URL", "https://api.qnaigc.com/v1")
        self.model = os.environ.get("QNAIGC_MODEL")
        if not api_key:
            raise RuntimeError("QNAIGC_API_KEY is not set")
        if not self.model:
            raise RuntimeError("QNAIGC_MODEL is not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _request(
        self, prompt: str, image_paths: list[str], max_retries: int = 8
    ) -> str:
        content = [_image_content(path) for path in image_paths]
        content.append({"type": "text", "text": prompt})
        retryable_errors = (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise visual reasoning assistant.",
                        },
                        {"role": "user", "content": content},
                    ],
                )
                return _response_text(completion.choices[0].message.content)
            except retryable_errors as error:
                if attempt + 1 == max_retries:
                    raise
                delay = min(60, 5 * (2**attempt))
                print(
                    f"QnAIGC request failed with {type(error).__name__}; "
                    f"retrying in {delay}s ({attempt + 1}/{max_retries - 1})"
                )
                time.sleep(delay)

        raise RuntimeError("QnAIGC retry loop exited unexpectedly")

    def request_with_image(self, prompt: str, image_path: str) -> str:
        return self._request(prompt, [image_path])

    def request_with_images(
        self, prompt: str, image_paths: list[str], image_format: str = "jpg"
    ) -> str:
        del image_format
        return self._request(prompt, image_paths)
