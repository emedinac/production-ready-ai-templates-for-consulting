"""Test schema validation."""

from CHANGE_ME.configs.validate import load_and_validate_config


def test_schema():
    """Test config schema validation."""
    assert load_and_validate_config().experiment.name
