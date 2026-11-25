"""Local presentation assets and execution-driven processing state. No model logic."""
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from base64 import b64encode

ASSETS = Path(__file__).parent / "assets"
STAGES = (
    ("checking", "Checking uploaded images"),
    ("loading", "Loading the segmentation model"),
    ("enhancing", "Preparing accepted ultrasound images"),
    ("segmenting", "Segmenting lesion regions"),
    ("preparing", "Preparing masks and overlays"),
    ("complete", "Analysis complete"),
)


@dataclass
class ProcessingState:
    stage: str = "checking"
    status: str = "running"
    completed: list = field(default_factory=list)
    visited: list = field(default_factory=lambda: ["checking"])
    accepted: int | None = None
    skipped: int | None = None

    def advance(self, stage):
        if self.status != "running":
            raise ValueError("A finished analysis cannot advance")
        if stage not in dict(STAGES):
            raise ValueError(f"Unknown analysis stage: {stage}")
        if stage == self.stage:
            return
        if list(dict(STAGES)).index(stage) <= list(dict(STAGES)).index(self.stage):
            raise ValueError("Analysis stages must advance in execution order")
        self.completed.append(self.stage)
        self.visited.append(stage)
        self.stage = stage
        if stage == "complete":
            self.status = "complete"

    def finish(self, status):
        if status not in ("rejected", "failed"):
            raise ValueError("Unknown final state")
        if status == "rejected" and self.stage not in self.completed:
            self.completed.append(self.stage)
        self.status = status


def icon():
    # st.html sanitizes inline SVG; a local data image preserves the icon safely.
    svg = (ASSETS / "scan-line.svg").read_text(encoding="utf-8").replace('currentColor', '#79e9da')
    return svg_image(svg)


def svg_image(svg):
    encoded = b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{encoded}" alt="" aria-hidden="true">'


def processing_html(state):
    active = state.status == "running"
    headline = dict(STAGES)[state.stage]
    if state.status == "rejected":
        headline = "No images passed the ultrasound input check"
    elif state.status == "failed":
        headline = "Analysis stopped"
    steps = []
    for index, (key, label) in enumerate(STAGES, 1):
        kind, description = "pending", "Pending"
        if key in state.completed or (key == "complete" and state.status == "complete"):
            kind, description = "done", "Completed"
        elif key == state.stage:
            kind = "active" if active else "stopped"
            description = "In progress" if active else "Stopped"
        elif key == "loading" and "enhancing" in state.visited and key not in state.visited:
            kind, description = "cached", "Already cached"
        elif not active:
            description = "Not run"
        steps.append(f'<li class="{kind}"><span class="g17-step-number">{index:02}</span>'
                     f'<span>{escape(label)}<small>{description}</small></span></li>')
    counts = "Original images are checked before segmentation."
    if state.accepted is not None:
        counts = f"{state.accepted} accepted · {state.skipped} skipped"
    waveform = svg_image('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 100"><path fill="none" stroke="#6adacc" stroke-width="1.5" d="M0 64H44L55 50L66 70L83 30L97 77L110 56H145L158 43L172 63H240"/></svg>')
    timeline = f'<ol class="g17-stepper">{"".join(steps)}</ol>'
    if not active:
        timeline = f'<details class="g17-timeline"><summary>View processing steps</summary>{timeline}</details>'
    activity = ('<div class="g17-indeterminate" aria-hidden="true"><i></i></div>'
                '<small>Work is in progress. Animation illustrates activity, not predictions.</small>') if active else ''
    return f'''<section class="g17-process {'is-running' if active else 'is-finished'}"
        data-stage="{escape(state.stage)}" data-status="{escape(state.status)}"
        aria-label="Analysis progress" aria-busy="{str(active).lower()}">
      <div class="g17-process-top"><div class="g17-scan" aria-hidden="true">
        <div class="g17-sweep"></div>{waveform}
        <span>ILLUSTRATIVE SCAN</span></div><div class="g17-process-copy">
        <div class="g17-eyebrow">ANALYSIS WORKSPACE</div>
        <h3 role="status" aria-live="polite">{escape(headline)}</h3>
        <p>{escape(counts)}</p>{activity}
      </div></div>{timeline}</section>'''


def header_html():
    return f'''<header class="g17-header"><div class="g17-brand-icon">{icon()}</div>
      <div class="g17-brand"><strong>G17</strong><span>Breast Ultrasound Segmentation</span></div>
      <span class="g17-badge"><i></i> RESEARCH DEMO</span></header>'''


def overview_html(accepted, skipped, processed):
    return '<div class="g17-overview">' + ''.join(
        f'<div><span>{label}</span><strong>{count:02}</strong><small>{detail}</small></div>'
        for label, count, detail in (("Accepted", accepted, "Ultrasound appearance"),
                                    ("Skipped", skipped, "Review reasons below"),
                                    ("Processed", processed, "Segmentation complete" if processed else "No segmentation results"))) + '</div>'
