"""Tests for misfit.models.model_registry."""
import pytest

from misfit.models.model_registry import (
    MODEL_REGISTRY,
    get_model_from_registry,
    list_registered_models,
    register_model,
)


def test_register_and_retrieve():
    name = "_test_model_reg_abc"
    # Clean up in case of reruns.
    MODEL_REGISTRY.pop(name, None)

    @register_model(name)
    def _factory(**kwargs):
        return "my_model"

    assert get_model_from_registry(name) == "my_model"
    MODEL_REGISTRY.pop(name, None)


def test_register_duplicate_raises():
    name = "_test_dup_model"
    MODEL_REGISTRY.pop(name, None)

    @register_model(name)
    def _f(**kwargs):
        return None

    with pytest.raises(ValueError, match="already registered"):
        @register_model(name)
        def _g(**kwargs):
            return None

    MODEL_REGISTRY.pop(name, None)


def test_get_unknown_raises():
    with pytest.raises(ValueError, match="not registered"):
        get_model_from_registry("totally_unknown_model_xyz")


def test_list_registered_models_sorted():
    models = list_registered_models()
    assert models == sorted(models)


def test_list_registered_models_contains_swinmae():
    import misfit.models  # noqa — trigger registrations
    models = list_registered_models()
    assert "swinmae-small" in models
    assert "swinmae-base" in models
    assert "swinmae-large" in models


def test_registry_kwargs_passed():
    name = "_test_kwargs_model"
    MODEL_REGISTRY.pop(name, None)

    @register_model(name)
    def _factory(**kwargs):
        return kwargs

    result = get_model_from_registry(name, foo=1, bar=2)
    assert result == {"foo": 1, "bar": 2}
    MODEL_REGISTRY.pop(name, None)
