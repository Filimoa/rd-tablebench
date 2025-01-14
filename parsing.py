"""
For each format, this code extracts the largest HTML table from the response.
"""

import json
from typing import Any
import os

import argparse
import glob


def parse_textract_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    return data["html_table"], data


def parse_gcloud_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, None

    return data["html_table"], data


def parse_reducto_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    if "error" in data:
        return None, data

    longest_html = None
    max_length = 0

    for chunk in data["result"]["chunks"]:
        blocks = chunk["blocks"]
        for block in blocks:
            if block["type"] == "Table":
                if len(block["content"]) > max_length:
                    max_length = len(block["content"])
                    longest_html = block["content"]

    return longest_html, data


def parse_chunkr_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    if data.get("status") != "Succeeded":
        return None, data

    largest_html = None
    max_length = 0

    try:
        for output in (
            data.get("output", [])
            if "chunks" not in data.get("output")
            else data["output"]["chunks"]
        ):
            for segment in output.get("segments", []):
                if segment.get("segment_type") == "Table" and segment.get("html"):
                    if len(segment["html"]) > max_length:
                        max_length = len(segment["html"])
                        largest_html = segment["html"]
    except Exception:
        import traceback

        traceback.print_exc()
        print(data)

    return largest_html, data


def parse_unstructured_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    largest_html = None
    max_length = 0

    for element in data.get("elements", []):
        if element.get("type") == "Table" and element.get("metadata", {}).get(
            "text_as_html"
        ):
            html = element["metadata"]["text_as_html"]
            if len(html) > max_length:
                max_length = len(html)
                largest_html = html

    return largest_html, data


def parse_gpt4o_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    html = data["html_table"]
    # Extract just the table portion between <table> and </table>
    start = html.find("<table>")
    end = html.find("</table>") + 8
    if start != -1 and end != -1:
        return html[start:end], data
    return None, data


def parse_gemini_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    html = data["html_table"]
    # Extract just the table portion between <table> and </table>
    start = html.find("<table>")
    end = html.find("</table>") + 8
    if start != -1 and end != -1:
        return html[start:end], data
    return None, data


def parse_azure_response(path: str) -> tuple[str | None, Any]:
    data = None
    try:
        with open(path, "r") as f:
            data = json.load(f)

        def azure_to_html(table: Any) -> str:
            html = "<table>"
            for row_index in range(table["rowCount"]):
                html += "<tr>"
                for col_index in range(table["columnCount"]):
                    cell = next(
                        (
                            c
                            for c in table["cells"]
                            if c["rowIndex"] == row_index
                            and c["columnIndex"] == col_index
                        ),
                        None,
                    )
                    if cell:
                        content = (
                            cell["content"]
                            .replace(":selected:", "")
                            .replace(":unselected:", "")
                        )
                        tag = "th" if cell.get("kind") == "columnHeader" else "td"
                        rowspan = (
                            f" rowspan='{cell['rowSpan']}'" if "rowSpan" in cell else ""
                        )
                        colspan = (
                            f" colspan='{cell['columnSpan']}'"
                            if "columnSpan" in cell
                            else ""
                        )
                        html += f"<{tag}{rowspan}{colspan}>{content}</{tag}>"
                    else:
                        pass
                html += "</tr>"
            html += "</table>"
            return html

        # Find table with largest area (row count * column count)
        largest_table = max(
            data["tables"], key=lambda t: t["rowCount"] * t["columnCount"]
        )
        return azure_to_html(largest_table), data
    except Exception:
        return None, data


PARSERS = {
    "textract": parse_textract_response,
    "gcloud": parse_gcloud_response,
    "reducto": parse_reducto_response,
    "chunkr": parse_chunkr_response,
    "unstructured": parse_unstructured_response,
    "gpt4o": parse_gpt4o_response,
    "azure": parse_azure_response,
    "gemini": parse_gemini_response,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        required=True,
        choices=PARSERS.keys(),
        help="Which parser to use (e.g. 'gpt4o', 'azure', etc.).",
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing .json files from the provider.",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder to write the extracted .html files.",
    )
    args = parser.parse_args()

    # Get the parser function based on the provider
    parse_func = PARSERS[args.provider]

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Find all JSON files under input folder (recursively)
    json_paths = glob.glob(
        os.path.join(args.input_folder, "**", "*.json"), recursive=True
    )

    for json_file in json_paths:
        # Parse the JSON to get HTML
        html, raw_data = parse_func(json_file)

        if not html:
            # No table found or parse error
            print(f"Skipping (no HTML found): {json_file}")
            continue

        # Build output path: replace .json with .html and replicate subfolders if desired
        relative_path = os.path.relpath(json_file, start=args.input_folder)
        out_name = os.path.splitext(relative_path)[0] + ".html"
        out_path = os.path.join(args.output_folder, out_name)

        # Make sure subdirectories exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Write the HTML to file
        with open(out_path, "w") as f:
            f.write(html)

        print(f"Saved HTML to: {out_path}")


if __name__ == "__main__":
    main()
