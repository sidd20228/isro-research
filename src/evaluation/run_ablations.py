from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.evaluation.ablation import default_ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the controlled V1 ablation matrix.")
    parser.add_argument("--output", default="reports/artifacts/ablation_manifest.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{"ablation_id": f"ablation_{index:02d}", **asdict(ablation)} for index, ablation in enumerate(default_ablations())])
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    main()
