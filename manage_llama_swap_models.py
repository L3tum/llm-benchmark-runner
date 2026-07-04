#!/usr/bin/env python3
"""
Helper to add a model configuration to llama-swap's config file.
This avoids manual editing and allows scriptable model registration.

Usage:
  python manage_llama_swap_models.py add --name mymodel-q4 --display "MyModel Q4" --backend "llama-server -m /path/to/model.gguf"
  python manage_llama_swap_models.py list [--config path]
  python manage_llama_swap_models.py remove --name mymodel-q4
"""

import argparse
import json
import os
import sys
from pathlib import Path


LLAMA_SWAP_CONFIG_PATH = Path.home() / ".config" / "llama-swap" / "config.json"


def load_config(config_path):
    """Load llama-swap config file."""
    if not config_path.exists():
        return {"models": []}
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config, config_path):
    """Save llama-swap config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def add_model(config_path, name, display, backend, extra_args=None):
    """Add a model to the config."""
    config = load_config(config_path)
    models = config.get("models", [])

    # Check if model already exists
    for m in models:
        if m.get("name") == name:
            print(f"Model '{name}' already exists. Updating...")
            m["name"] = name
            m["display"] = display
            m["backend"] = backend
            if extra_args:
                m.update(extra_args)
            break
    else:
        model_entry = {
            "name": name,
            "display": display,
            "backend": backend,
        }
        if extra_args:
            model_entry.update(extra_args)
        models.append(model_entry)

    config["models"] = models
    save_config(config, config_path)
    print(f"Model '{name}' added/updated in {config_path}")


def remove_model(config_path, name):
    """Remove a model from the config."""
    config = load_config(config_path)
    models = config.get("models", [])
    new_models = [m for m in models if m.get("name") != name]
    if len(new_models) == len(models):
        print(f"Model '{name}' not found.")
        return
    config["models"] = new_models
    save_config(config, config_path)
    print(f"Model '{name}' removed from {config_path}")


def list_models(config_path):
    """List all models in the config."""
    config = load_config(config_path)
    models = config.get("models", [])
    if not models:
        print("No models configured.")
        return
    print(f"Models in {config_path}:\n")
    for i, m in enumerate(models):
        name = m.get("name", "?")
        display = m.get("display", "?")
        backend = m.get("backend", "?")
        print(f"{i+1}. {name} ({display})")
        print(f"   Backend: {backend}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Manage llama-swap model configurations.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add model
    add_parser = subparsers.add_parser("add", help="Add or update a model")
    add_parser.add_argument("--name", required=True, help="Model name (used in API)")
    add_parser.add_argument("--display", required=True, help="Display name")
    add_parser.add_argument("--backend", required=True, help="Backend command (e.g., llama-server -m /path/to/model.gguf)")
    add_parser.add_argument("--extra-args", nargs="*", help="Extra arguments (key=value pairs)")
    add_parser.add_argument("--config", default=LLAMA_SWAP_CONFIG_PATH, type=Path, help="Path to config file")

    # Remove model
    remove_parser = subparsers.add_parser("remove", help="Remove a model")
    remove_parser.add_argument("--name", required=True, help="Model name")
    remove_parser.add_argument("--config", default=LLAMA_SWAP_CONFIG_PATH, type=Path, help="Path to config file")

    # List models
    list_parser = subparsers.add_parser("list", help="List all models")
    list_parser.add_argument("--config", default=LLAMA_SWAP_CONFIG_PATH, type=Path, help="Path to config file")

    args = parser.parse_args()

    if args.command == "add":
        extra = {}
        if args.extra_args:
            for arg in args.extra_args:
                if "=" in arg:
                    k, v = arg.split("=", 1)
                    extra[k] = v
        add_model(args.config, args.name, args.display, args.backend, extra)
    elif args.command == "remove":
        remove_model(args.config, args.name)
    elif args.command == "list":
        list_models(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
