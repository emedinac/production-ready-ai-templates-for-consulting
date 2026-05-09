import os
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


def wait_for_mlflow(timeout: int = 120, interval: int = 2) -> None:
    base_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    tracking_uri = f"{base_uri.rstrip('/')}/health"
    deadline = time.time() + timeout
    print(f"Waiting for MLflow at {tracking_uri}...")

    while time.time() < deadline:
        try:
            response = urlopen(tracking_uri, timeout=5)
            print(f"MLflow is available: {response.status}")
            return
        except (HTTPError, URLError) as exc:
            print(f"MLflow not available yet: {exc}")
            time.sleep(interval)

    raise SystemExit(f"MLflow did not become available within {timeout} seconds")


if __name__ == "__main__":
    wait_for_mlflow()
