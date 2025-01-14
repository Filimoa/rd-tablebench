import os
import glob

from convert import html_to_numpy
from grading import table_similarity
import polars as pl
import argparse

from providers.config import settings


models = {
    "gemini-2.0-flash-exp": {
        "folder": settings.output_dir / "gemini-2.0-flash-exp-raw",
    },
    "gemini-1.5-flash": {
        "folder": settings.output_dir / "gemini-1.5-flash-raw",
    },
    "gemini-1.5-pro": {
        "folder": settings.output_dir / "gemini-1.5-pro-raw",
    },
    "gpt-4o": {
        "folder": settings.output_dir / "gpt-4o-raw",
    },
    "gpt-4o-mini": {
        "folder": settings.output_dir / "gpt-4o-mini-raw",
    },
    "claude-3-5-sonnet": {
        "folder": settings.output_dir / "claude-3-5-sonnet-latest-raw",
    },
    "reducto": {
        "folder": settings.output_dir / "reducto",
    },
}


def main(model: str, folder: str):
    groundtruth = "/Users/sergey/Downloads/rd-tablebench/groundtruth"
    # Prepare somewhere to store results
    scores = []

    # Loop over each .html file in the GPT-4o folder
    html_files = glob.glob(os.path.join(folder, "*.html"))
    for pred_html_path in html_files:
        # The filename might be something like: 10035_png.rf.07e8e5bf2e9ad4e77a84fd38d1f53f38.html
        base_name = os.path.basename(pred_html_path)

        # Build the path to the corresponding ground-truth file
        gt_html_path = os.path.join(groundtruth, base_name)
        if not os.path.exists(gt_html_path):
            # If no matching ground-truth HTML, skip
            continue

        # Read the prediction HTML
        with open(pred_html_path, "r") as f:
            pred_html = f.read()

        # Read the ground-truth HTML
        with open(gt_html_path, "r") as f:
            gt_html = f.read()

        # Convert HTML -> NumPy arrays
        try:
            pred_array = html_to_numpy(pred_html)
            gt_array = html_to_numpy(gt_html)
            # Compute similarity (0.0 to 1.0)
            score = table_similarity(gt_array, pred_array)
        except Exception as e:
            print(f"Error converting {base_name}: {e}")
            continue

        # Store or print the result
        scores.append((base_name, score))
        print(f"{base_name}: {score:.4f}")

    score_dicts = [{"filename": fname, "score": scr} for fname, scr in scores]
    df = pl.DataFrame(score_dicts)
    print(
        f"Average score for {model}: {df['score'].mean():.4f} with std {df['score'].std():.4f}"
    )
    df.write_csv(f"./scores/{model}_scores.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.model, models[args.model]["folder"])
