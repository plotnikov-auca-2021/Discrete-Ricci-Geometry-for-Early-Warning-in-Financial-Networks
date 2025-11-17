"""
Entry point for running the full walk-forward experiment.

At this stage, the core pipeline is only scaffolded. Once the modules
(graphs, curvature, features, labels, models, pipeline) are implemented,
you can run:

    python -m scripts.run_walkforward

from the project root.
"""

from ricci_ews import config
from ricci_ews import pipeline


def main():
    print("Running walk-forward experiment (skeleton)...")
    try:
        results = pipeline.not_implemented()
    except NotImplementedError as e:
        print("Pipeline not implemented yet:", e)
        return
    print(results.head())


if __name__ == "__main__":
    main()
