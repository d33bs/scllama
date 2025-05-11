"""
Utils for working with llama.cpp in Python.
"""

import requests
from IPython.display import display, HTML
import markdown

from typing import Dict, Any
from IPython.display import display, HTML
import markdown


def query_llama_with_image_path(
    image_path: str,
    prompt: str,
    server_url: str = "http://localhost:8080/v1/completions"
) -> Dict:
    """
    Send an image-aware prompt to a local llama.cpp server using file-based image reference.

    This function constructs a prompt in the format `/image path.jpg\nYour prompt` and sends it
    to the `/v1/completions` endpoint, assuming that the model supports vision input via
    image path syntax.

    Args:
        image_path (str): Path to the image file on disk. Must be accessible by the server.
        prompt (str): The text prompt to accompany the image.
        server_url (str, optional): The local llama.cpp completion endpoint. Defaults to
            "http://localhost:8080/v1/completions".

    Returns:
        Dict: The parsed JSON response from the server, typically containing a "completion" field.

    Raises:
        requests.HTTPError: If the HTTP request fails or returns a non-200 status.
    """
    full_prompt = f"/image {image_path}\n{prompt}"

    payload = {
        "prompt": full_prompt,
        "temperature": 0.7,
        "max_tokens": 512,
        "stop": None,
        "stream": False
    }

    response = requests.post(server_url, json=payload)
    response.raise_for_status()
    return response.json()



def display_response(response: Dict[str, Any]) -> None:
    """
    Render and display Markdown-formatted text
    from a model response in a Jupyter notebook.

    This function handles both OpenAI-style
    completions (`text`) and chat-style messages 
    (`message["content"]`), supporting multi-part
    responses by concatenating all returned segments.

    Args:
        response (Dict[str, Any]): 
            A dictionary containing the model output.
            Expected structure:
                {
                    "choices": [
                        {"text": "..."},
                        {"message": {"content": "..."}},
                        ...
                    ]
                }

    Raises:
        ValueError:
            If the response does not contain recognizable 
            `text` or `message["content"]` fields.
    """
    try:
        parts = []
        for choice in response["choices"]:
            if "text" in choice:
                parts.append(choice["text"])
            elif "message" in choice and "content" in choice["message"]:
                parts.append(choice["message"]["content"])
            else:
                raise ValueError("No recognizable text or message content in response choice")

        full_markdown = "\n\n".join(parts)
        html = markdown.markdown(full_markdown)
        display(HTML(html))
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("Unexpected response format") from e
