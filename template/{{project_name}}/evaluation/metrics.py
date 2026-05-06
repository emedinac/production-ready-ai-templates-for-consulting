from pathlib import Path
import json


def write_evaluation_metrics() -> Path:
    project_root = Path(__file__).resolve().parents[1]

    output_path = project_root / "results/eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps({"accuracy": 0.0, "f1": 0.0}, indent=2))

    return output_path


if __name__ == "__main__":
    write_evaluation_metrics()
