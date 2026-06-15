"""Lightweight YAML configuration with dot-access and deep merge.

Configs support an optional ``__base__`` key (str or list of str) pointing to
parent YAML files (relative to the current file). Child values override parents.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Union

import yaml


class Config(dict):
    """A dict subclass allowing attribute-style access (``cfg.train.lr``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def get(self, key: str, default: Any = None) -> Any:
        value = super().get(key, default)
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.items():
            out[k] = v.to_dict() if isinstance(v, Config) else copy.deepcopy(v)
        return out


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping.")
    return data


def load_config(path: str) -> Config:
    """Load a YAML config resolving ``__base__`` inheritance."""
    path = os.path.abspath(path)
    raw = _load_yaml(path)

    bases: Union[str, List[str]] = raw.pop("__base__", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: Dict[str, Any] = {}
    for base_rel in bases:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_rel))
        merged = merge_configs(merged, dict(load_config(base_path)))

    merged = merge_configs(merged, raw)
    return Config(merged)


def override_from_dotlist(cfg: Config, dotlist: List[str]) -> Config:
    """Apply CLI overrides of the form ``train.lr=1e-4`` onto ``cfg``."""
    for item in dotlist:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be key=value.")
        key, value = item.split("=", 1)
        node: Dict[str, Any] = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, Config())
        node[parts[-1]] = yaml.safe_load(value)
    return cfg
