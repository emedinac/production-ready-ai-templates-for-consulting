"""DVC pipeline helpers for production ML workflows."""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


def run_stage(stage_name: str, force: bool = False) -> bool:
    """Run a DVC stage by name.

    Args:
        stage_name: Name of the DVC stage to run
        force: Force re-run even if dependencies haven't changed

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["dvc", "repro", stage_name]
        if force:
            cmd.append("--force")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully ran stage: {stage_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to run stage {stage_name}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def run_pipeline(force: bool = False) -> bool:
    """Run the entire DVC pipeline.

    Args:
        force: Force re-run of all stages

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["dvc", "repro"]
        if force:
            cmd.append("--force")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Successfully ran full pipeline")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to run pipeline: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def list_stages() -> List[str]:
    """List all available pipeline stages from dvc.yaml.

    Returns:
        List of stage names
    """
    try:
        with open("dvc.yaml", "r") as f:
            dvc_config = yaml.safe_load(f)
        return list(dvc_config.get("stages", {}).keys())
    except (FileNotFoundError, yaml.YAMLError):
        # Fallback to hardcoded stages
        return ["preprocess", "train", "evaluate"]


def get_stage_status(stage_name: str) -> Dict[str, Any]:
    """Get the status of a DVC stage.

    Args:
        stage_name: Name of the stage to check

    Returns:
        Dictionary with stage status information
    """
    try:
        result = subprocess.run(
            ["dvc", "status", stage_name], capture_output=True, text=True, check=True
        )
        return {"status": "completed", "output": result.stdout}
    except subprocess.CalledProcessError:
        return {"status": "pending", "output": ""}


def push_data(remote: str = "data") -> bool:
    """Push data artifacts to remote storage.

    Args:
        remote: Name of the DVC remote

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(["dvc", "push", "--remote", remote], check=True)
        print(f"Successfully pushed data to remote: {remote}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to push data: {e}")
        return False


def pull_data(remote: str = "data") -> bool:
    """Pull data artifacts from remote storage.

    Args:
        remote: Name of the DVC remote

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(["dvc", "pull", "--remote", remote], check=True)
        print(f"Successfully pulled data from remote: {remote}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull data: {e}")
        return False


def get_metrics(stage_name: str) -> Optional[Dict[str, Any]]:
    """Get metrics from a completed stage.

    Args:
        stage_name: Name of the stage

    Returns:
        Dictionary of metrics or None if not found
    """
    # Try to find metrics file based on stage
    metrics_paths = {
        "preprocess": "{{project_name}}/data/preprocess_metrics.json",
        "train": "{{project_name}}/models/metrics.json",
        "evaluate": "results/eval_metrics.json",
    }

    metrics_file = metrics_paths.get(stage_name)
    if metrics_file and Path(metrics_file).exists():
        try:
            with open(metrics_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def validate_pipeline() -> List[str]:
    """Validate the DVC pipeline configuration.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check if dvc.yaml exists
    if not Path("dvc.yaml").exists():
        errors.append("dvc.yaml not found")
        return errors

    # Check if DVC is initialized
    if not Path(".dvc").exists():
        errors.append("DVC not initialized (.dvc directory missing)")

    # Validate stage dependencies
    try:
        with open("dvc.yaml", "r") as f:
            dvc_config = yaml.safe_load(f)

        stages = dvc_config.get("stages", {})
        for stage_name, stage_config in stages.items():
            # Check if command exists
            if "cmd" not in stage_config:
                errors.append(f"Stage '{stage_name}' missing cmd")

            # Check if dependencies exist
            deps = stage_config.get("deps", [])
            for dep in deps:
                if not Path(dep).exists() and not dep.startswith("{{"):
                    errors.append(
                        f"Dependency '{dep}' not found for stage '{stage_name}'"
                    )

    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in dvc.yaml: {e}")

    return errors


def detect_storage_backend() -> str:
    """Detect the configured storage backend from DVC config.

    Returns:
        's3' or 'postgresql' or 'unknown'
    """
    try:
        with open(".dvc/config", "r") as f:
            content = f.read()
            if "s3://" in content:
                return "s3"
            elif "postgresql://" in content:
                return "postgresql"
    except FileNotFoundError:
        pass
    return "unknown"


def setup_storage_backend(backend: str, **kwargs) -> bool:
    """Set up DVC remotes for the specified backend.

    Args:
        backend: 's3' or 'postgresql'
        **kwargs: Backend-specific configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        if backend == "s3":
            bucket_cache = kwargs.get(
                "bucket_cache", "your-production-bucket/dvc-cache"
            )
            bucket_models = kwargs.get("bucket_models", "your-models-bucket/models")
            bucket_data = kwargs.get("bucket_data", "your-data-bucket/data")

            subprocess.run(
                ["dvc", "remote", "add", "-d", "storage", f"s3://{bucket_cache}"],
                check=True,
            )
            subprocess.run(
                ["dvc", "remote", "add", "models", f"s3://{bucket_models}"], check=True
            )
            subprocess.run(
                ["dvc", "remote", "add", "data", f"s3://{bucket_data}"], check=True
            )

        elif backend == "postgresql":
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", "5432")
            user = kwargs.get("user", "postgres")
            password = kwargs.get("password", "password")

            base_url = f"postgresql://{user}:{password}@{host}:{port}"

            subprocess.run(
                ["dvc", "remote", "add", "-d", "storage", f"{base_url}/dvc_cache"],
                check=True,
            )
            subprocess.run(
                ["dvc", "remote", "add", "models", f"{base_url}/models"], check=True
            )
            subprocess.run(
                ["dvc", "remote", "add", "data", f"{base_url}/data"], check=True
            )

        else:
            print(f"Unsupported backend: {backend}")
            return False

        print(f"Successfully configured {backend} storage backend")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to setup {backend} backend: {e}")
        return False
