"""OSBS inference entry point (replaces the old multi-checkpoint batch script).

Run from the repository root (or pass an absolute path to ``--config``):

    python predict.py
    python predict.py --config /path/to/config.yml

Equivalent module invocation:

    python -m src.pipelines.osbs_inference --config config.yml
"""

from src.pipelines.osbs_inference import main

if __name__ == "__main__":
    main()
