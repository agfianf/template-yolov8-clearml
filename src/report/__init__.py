"""Interactive, self-contained HTML evaluation report for a finished training run.

`build_evaluation_report` is the only thing outside this package should need. It runs
once, after the final `val()`, and uploads a single self-contained HTML file as a
ClearML artifact -- plus a link to it in Debug Samples, because the artifact panel is
not where anyone looks first.
"""

from src.report.build import build_evaluation_report


__all__ = ["build_evaluation_report"]
