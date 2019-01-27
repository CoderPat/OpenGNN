"""Defines functions related to configuration files."""

from importlib import import_module

import sys
import io
import os
from typing import List, Optional, Dict, Any

import yaml

from opengnn.models import catalog


def load_model_module(path):
    """Loads a model configuration file.

    Args:
        path: The relative path to the configuration file.

    Returns:
        A Python module.
    """
    dirname, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, os.path.abspath(dirname))
    module = import_module(module_name)

    if not hasattr(module, "model"):
        raise ImportError("No model defined in {}".format(path))

    return module


def load_subtokenizer(path):
    dirname, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, os.path.abspath(dirname))
    module = import_module(module_name)

    if not hasattr(module, "subtokenizer"):
        raise ImportError("No subtokenizer defined in {}".format(path))

    return module.subtokenizer


def load_model_from_catalog(name):
    """Loads a model from the catalog.

    Args:
        name: The model name.

    Returns:
        A :class:`opengnn.models.model.Model` instance.
    """
    return getattr(catalog, name)()


def load_model(model_dir,
               model_file=None,
               model_name=None):
    """Loads the model from the catalog or a file.

    The model object is pickled in :obj:`model_dir` to make the model
    configuration optional for future runs.

    Args:
        model_dir: The model directory.
        model_file: An optional model configuration.
            Mutually exclusive with :obj:`model_name`.
        model_name: An optional model name from the catalog.
            Mutually exclusive with :obj:`model_file`.

    Returns:
        A :class:`opengnn.models.model.Model` instance.

    Raises:
        ValueError: if both :obj:`model_file` and :obj:`model_name` are set.
    """
    if model_file and model_name:
        raise ValueError("only one of model_file and model_name should be set")

    if not model_file and not model_name:
        raise ValueError("either model_file and model_name needs to be set")

    if model_file:
        model_config = load_model_module(model_file)
        model = model_config.model()
    elif model_name:
        model = load_model_from_catalog(model_name)

    return model


def load_config(config_paths: List[str], config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Loads configuration files.
    Args:
        config_paths: A list of configuration file paths.
        config: A (possibly non empty) config dictionary to fill.
    Returns:
        The configuration dictionary.
    """
    if config is None:
        config = {}

    for config_path in config_paths:
        with io.open(config_path, encoding="utf-8") as config_file:
            subconfig = yaml.load(config_file.read())
            # Add or update section in main configuration.
            for section in subconfig:
                if section in config:
                    if isinstance(config[section], dict):
                        config[section].update(subconfig[section])
                    else:
                        config[section] = subconfig[section]
                else:
                    config[section] = subconfig[section]

    return config
