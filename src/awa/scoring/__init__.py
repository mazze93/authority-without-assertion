"""The scorer: pure functions from (scenario ground truth, annotation) to a classification."""
from .score import (Classification, AnnotationError, load_rule, score, fapr, summarize,
                    validate_annotation)
from .preannotate import preannotate

__all__ = ["Classification", "AnnotationError", "load_rule", "score", "fapr", "summarize",
           "validate_annotation", "preannotate"]
