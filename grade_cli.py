import os
import glob
import argparse

from convert import html_to_numpy
from grading import table_similarity
import polars as pl

from providers.config import settings


def main(model: str, folder: str, save_to_csv: bool):
    groundtruth = settings.input_dir / "groundtruth"
    scores = []

    html_files = glob.glob(os.path.join(folder, "*.html"))
    for pred_html_path in html_files:
        # The filename might be something like: 10035_png.rf.07e8e5bf2e9ad4e77a84fd38d1f53f38.html
        base_name = os.path.basename(pred_html_path)

        # Build the path to the corresponding ground-truth file
        gt_html_path = os.path.join(groundtruth, base_name)
        if not os.path.exists(gt_html_path):
            continue

        with open(pred_html_path, "r") as f:
            pred_html = f.read()

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

        scores.append((base_name, score))
        print(f"{base_name}: {score:.4f}")

    score_dicts = [{"filename": fname, "score": scr} for fname, scr in scores]
    df = pl.DataFrame(score_dicts)
    print(
        f"Average score for {model}: {df['score'].mean():.2f} with std {df['score'].std():.2f}"
    )
    if save_to_csv:
        df.write_csv(f"./scores/{model}_scores.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save-to-csv", type=bool, default=True)
    args = parser.parse_args()

    model_dir = settings.output_dir / f"{args.model}-raw"
    assert model_dir.exists(), f"Model directory {model_dir} does not exist"
    main(args.model, model_dir, args.save_to_csv)
