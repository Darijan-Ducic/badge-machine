"""Ask Muse Spark (Meta Model API) a question about an mp4 video — public URL or local file."""

import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"
# Inline files are capped at 50 MB; checked against the base64 size to stay under it either way.
INLINE_LIMIT = 50_000_000


def video_block(client: OpenAI, source: str) -> dict:
    if source.startswith(("http://", "https://")):
        return {"type": "input_video", "video_url": source}
    path = Path(source)
    if 4 * ((path.stat().st_size + 2) // 3) > INLINE_LIMIT:
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="user_data")
        return {"type": "input_video", "file_id": uploaded.id}
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "input_video", "video_url": f"data:video/mp4;base64,{data}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="mp4 file path or public URL")
    parser.add_argument("prompt", nargs="?", default="Describe what happens in this video.")
    args = parser.parse_args()

    # The OpenAI SDK does not auto-read MODEL_API_KEY, so pass it explicitly.
    client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"], timeout=600)
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": args.prompt},
                    video_block(client, args.video),
                ],
            }
        ],
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
