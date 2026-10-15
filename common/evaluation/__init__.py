"""Shared evaluation utilities; importing this package makes no external calls."""

__all__ = [
    "EvaluationExample", "load_examples", "sync_dataset",
    "extract_final_answer", "make_target", "run_experiment",
]


def __getattr__(name):
    # Keep ``python -m common.evaluation.dataset`` free of runpy re-imports.
    from importlib import import_module
    if name not in __all__:
        raise AttributeError(name)
    module = "dataset" if name in {"EvaluationExample", "load_examples", "sync_dataset"} else "runner"
    return getattr(import_module(f"{__name__}.{module}"), name)
