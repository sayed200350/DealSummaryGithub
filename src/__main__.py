"""
Entry point for running the Deal Summary & QA Bot as a module.

This allows the package to be executed with:
python -m src
"""

from .cli import main

if __name__ == "__main__":
    main()