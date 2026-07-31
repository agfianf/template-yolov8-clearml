"""Re-render the evaluation report locally, in about a second, without training.

Two sources, for two different kinds of change:

`replay` takes a report that a real run already published -- a local file or the
artifact URL -- and rebuilds the page around its data with the code in the working
tree. The blob a published page carries is the whole report minus the figures, which
`render.py` strips because the SVG is already in the markup; this tool lifts those SVG
elements back out of the old document and hands them to the renderer as if they had
just been drawn. Everything downstream of the blob is therefore live: the template,
the stylesheet, the script, section order and wording, captions, the TOC, highlights.
Only the figure *geometry* is frozen, because the numbers behind it were never
serialised.

`fixture` drives the real pipeline end to end over the synthetic dataset the report
tests use, so `figures.py` and `blob.py` changes show up. The numbers are invented.

    uv run tools/report_preview.py replay <file-or-url> [-o out.html]
    uv run tools/report_preview.py fixture [--seg] [--classes N] [-o out.html]

Reach for `replay` for anything about how the page reads or behaves, and for `fixture`
when a chart itself has to be redrawn.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import tempfile
import urllib.request

from pathlib import Path

from src.report.render import dom_estimate, render_report
from src.utils.logging import get_logger


logger = get_logger(__name__)

BLOB = re.compile(
    r'<script type="application/json" id="report-data">(.*?)</script>', re.DOTALL
)
FIGURE = re.compile(r'<figure class="[^"]*" data-fig="([^"]+)">(.*?)</figure>', re.DOTALL)
SVG = re.compile(r"<svg\b.*</svg>", re.DOTALL)
DEFAULT_OUT = Path(tempfile.gettempdir()) / "report_preview.html"


def read_source(source: str) -> str:
    """Return the document text, whether `source` names a URL or a file."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def extract_blob(document: str) -> dict:
    """Pull the JSON payload back out of a rendered page."""
    match = BLOB.search(document)
    if not match:
        raise SystemExit("no report-data element: this is not a rendered report")
    return json.loads(match.group(1).replace("<\\/", "</"))


def extract_figures(document: str) -> dict[str, dict]:
    """Lift each inlined `<svg>` back into the figure payload shape the renderer takes."""
    found: dict[str, dict] = {}
    for fig_id, body in FIGURE.findall(document):
        svg = SVG.search(body)
        if svg:
            found[fig_id] = {"svg": svg.group(0)}
        else:
            logger.warning("figure %s carries no svg; it will render as missing", fig_id)
    return found


def load_replay(source: str) -> dict:
    """Rebuild the blob that produced an already-published report."""
    document = read_source(source)
    blob = extract_blob(document)
    blob["figures"] = extract_figures(document)
    meta = blob.get("meta") or {}
    logger.info(
        "replaying %s (task %s, built by %s)",
        meta.get("run_name", "?"),
        str(meta.get("task_id", "?"))[:8],
        meta.get("code_version", "?"),
    )
    logger.info(
        "recovered %d figures and %d thumbnails from %.2f MB of markup",
        len(blob["figures"]),
        len(blob.get("thumbs") or {}),
        len(document.encode("utf-8")) / 1e6,
    )
    logger.warning(
        "figure geometry is frozen at the published build; use `fixture` for figures.py"
    )
    return blob


def load_fixture(*, seg: bool, classes: int) -> dict:
    """Build the report tests' synthetic blob through the real pipeline."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    path = root / "tests" / "report" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_report_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    directory = Path(tempfile.mkdtemp(prefix="report-preview-"))
    blob = module.make_blob(directory, seg=seg, n_classes=classes)
    logger.info(
        "built a %s fixture: %d classes, %d figures, %d thumbnails",
        "segment" if seg else "detect",
        classes,
        len(blob.get("figures") or {}),
        len(blob.get("thumbs") or {}),
    )
    return blob


def main() -> int:
    """Render one preview and log where it landed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    replay = sub.add_parser("replay", help="rebuild a published report's page")
    replay.add_argument("source", help="path or URL of a rendered evaluation_report.html")

    fixture = sub.add_parser("fixture", help="build one from synthetic data")
    fixture.add_argument("--seg", action="store_true", help="segmentation, not box")
    fixture.add_argument("--classes", type=int, default=8, help="how many classes")

    for name in ("replay", "fixture"):
        sub.choices[name].add_argument(
            "-o", "--out", type=Path, default=DEFAULT_OUT, help="where to write the page"
        )

    args = parser.parse_args()
    if args.mode == "replay":
        blob = load_replay(args.source)
    else:
        blob = load_fixture(seg=args.seg, classes=args.classes)

    document = render_report(blob)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")
    logger.info(
        "%.2f MB, %d DOM nodes -> file://%s",
        len(document.encode("utf-8")) / 1e6,
        dom_estimate(document),
        args.out.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
