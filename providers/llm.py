import argparse
import base64
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Literal

import backoff
import openai
from openai import OpenAI
from pdf2image import convert_from_path
from tqdm import tqdm

from providers.config import settings


base_path = os.path.expanduser(settings.input_dir)
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)


def convert_pdf_to_base64_image(pdf_path):
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    img_buffer = BytesIO()
    images[0].save(img_buffer, format="PNG")
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")


@backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=5)
def analyze_document_openai_sdk(base64_image, model: str):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert the image to an HTML table. The output should begin with <table> and end with </table>. Specify rowspan and colspan attributes when they are greater than 1. Do not specify any other attributes. Only use table related HTML tags, no additional formatting is required.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


@backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=5)
def analyze_document_anthropic(base64_image, model: str):
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert the image to an HTML table. The output should begin with <table> and end with </table>. Specify rowspan and colspan attributes when they are greater than 1. Do not specify any other attributes. Only use table related HTML tags, no additional formatting is required.",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    },
                ],
            }
        ],
    )
    return response.content[0].text


def parse_gemini_response(content: str) -> tuple[str | None, Any]:
    # Extract just the table portion between <table> and </table>
    start = content.find("<table>")
    end = content.find("</table>") + 8
    if start != -1 and end != -1:
        return content[start:end], None
    return None, None


def process_pdf(pdf_path: str, model: str):
    output_path = pdf_path.replace("pdfs", f"outputs/{model}-raw").replace(
        ".pdf", ".html"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        base64_image = convert_pdf_to_base64_image(pdf_path)
        if "gemini" in model:
            html_table = analyze_document_openai_sdk(base64_image, model)
        elif "gpt" in model:
            html_table = analyze_document_openai_sdk(base64_image, model)
        elif "claude" in model:
            html_table = analyze_document_anthropic(base64_image, model)
        else:
            raise ValueError(f"Unknown model: {model}")

        html, _ = parse_gemini_response(html_table)

        if not html:
            print(f"Skipping (no HTML found): {pdf_path}")
            return pdf_path, None

        with open(output_path, "w") as f:
            f.write(html_table)

        return pdf_path, None
    except Exception as e:
        return pdf_path, str(e)


def process_all_pdfs(
    pdfs: list[str],
    model: Literal[
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-5-sonnet-latest",
    ],
    max_workers: int,
):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf, model): pdf for pdf in pdfs}

        progress_bar = tqdm(total=len(pdfs), desc="Processing PDFs")

        for future in as_completed(futures):
            pdf_path, error = future.result()
            if error:
                print(f"Error processing {pdf_path}: {error}")
            progress_bar.update(1)

        progress_bar.close()

    print(f"Processed {len(pdfs)} PDFs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    if "gemini" in args.model:
        assert settings.gemini_api_key

        client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif "gpt" in args.model:
        assert settings.openai_api_key

        client = OpenAI(api_key=settings.openai_api_key)
    elif "claude" in args.model:
        from anthropic import Anthropic

        assert settings.anthropic_api_key
        client = Anthropic(api_key=settings.anthropic_api_key)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    process_all_pdfs(pdfs, args.model, args.num_workers)
