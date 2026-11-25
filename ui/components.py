"""
Reusable HTML snippets for the InsightIQ dashboard.

Every function returns an HTML string meant for
``st.markdown(..., unsafe_allow_html=True)``. Styling lives in
``styles/theme.css``; these helpers only emit markup + class names.
"""

from html import escape
from pathlib import Path

import streamlit as st

# ── Inline SVG icons (Material icons can't be used inside raw HTML blocks) ────
_SVG_PATHS = {
    "movie":    '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M7 4v16M17 4v16M2 9h5M2 15h5M17 9h5M17 15h5"/>',
    "layers":   '<path d="m12 2 10 5-10 5L2 7l10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "clock":    '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "hourglass":'<path d="M6 2h12M6 22h12M7 2c0 5 10 7 10 10S7 17 7 22M17 2c0 5-10 7-10 10s10 5 10 10"/>',
    "sparkles": '<path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    "link":     '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
    "audio":    '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',
    "mic":      '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v5"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chat":     '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "file":     '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/>',
    "upload":   '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/>',
    "scissors": '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.1 15.9M14.5 14.5 20 20M8.1 8.1 12 12"/>',
    "cpu":      '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
    "pin":      '<path d="M12 22s7-6.2 7-12a7 7 0 0 0-14 0c0 5.8 7 12 7 12z"/><circle cx="12" cy="10" r="2.5"/>',
    "tag":      '<path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
    "play":     '<path d="M7 4v16l13-8z"/>',
    "check":    '<path d="M20 6 9 17l-5-5"/>',
    "pages":    '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>',
    "timer":    '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2M9 2h6"/>',
}


def icon(name: str, size: int = 18, cls: str = "") -> str:
    """Return an inline stroke-style SVG icon that inherits currentColor."""
    path = _SVG_PATHS.get(name, _SVG_PATHS["sparkles"])
    return (
        f'<svg class="iq-icon {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )


# ── Global CSS loader ────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).resolve().parent.parent / "styles" / "theme.css"


def load_css() -> None:
    """Inject styles/theme.css once per script run."""
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


# ── Small atoms ──────────────────────────────────────────────────────────────
def badge(text: str, tone: str = "violet", icon_name: str | None = None) -> str:
    """Rounded pill badge. tone: violet | emerald | amber | slate | red."""
    ic = icon(icon_name, 13) if icon_name else ""
    return f'<span class="iq-badge iq-badge-{tone}">{ic}{escape(str(text))}</span>'


def status_pill(ok: bool, ok_text: str = "Ready", bad_text: str = "Not ready") -> str:
    tone = "ok" if ok else "bad"
    label = ok_text if ok else bad_text
    return f'<span class="iq-pill iq-pill-{tone}"><span class="iq-dot"></span>{escape(label)}</span>'


# ── Page headers ─────────────────────────────────────────────────────────────
def hero(title: str, subtitle: str, icon_name: str = "movie") -> str:
    """Compact home hero. CTA buttons are rendered by Streamlit right after it."""
    return f"""
<div class="iq-hero">
  <div class="iq-hero-icon">{icon(icon_name, 22)}</div>
  <div>
    <h1>{escape(title)}</h1>
    <p>{escape(subtitle)}</p>
  </div>
</div>"""


def page_header(title: str, subtitle: str, icon_name: str) -> str:
    """Slim header for inner pages (Process / Ask / History)."""
    return f"""
<div class="iq-page-header">
  <div class="iq-hero-icon">{icon(icon_name, 20)}</div>
  <div>
    <h2>{escape(title)}</h2>
    <p>{escape(subtitle)}</p>
  </div>
</div>"""


def section_title(text: str, icon_name: str | None = None) -> str:
    ic = icon(icon_name, 18) if icon_name else ""
    return f'<div class="iq-section-title">{ic}<span>{escape(text)}</span></div>'


# ── Stat cards ───────────────────────────────────────────────────────────────
def stat_grid(cards: list[dict]) -> str:
    """
    One CSS grid → every card gets equal height.
    card keys: icon, label, value (str) OR pill (html), context (optional), tone (optional)
    """
    items = []
    for c in cards:
        main = c["pill"] if c.get("pill") else f'<div class="iq-stat-value">{escape(str(c["value"]))}</div>'
        ctx = f'<div class="iq-stat-context">{escape(c["context"])}</div>' if c.get("context") else ""
        tone = c.get("tone", "violet")
        items.append(f"""
<div class="iq-card iq-stat">
  <div class="iq-stat-icon iq-tone-{tone}">{icon(c["icon"], 18)}</div>
  {main}
  <div class="iq-stat-label">{escape(c["label"])}</div>
  {ctx}
</div>""")
    return f'<div class="iq-stat-grid">{"".join(items)}</div>'


# ── How-it-works stepper ─────────────────────────────────────────────────────
def stepper(steps: list[tuple[str, str]]) -> str:
    """steps: list of (title, tool). Horizontal on wide screens, vertical on narrow."""
    parts = []
    for i, (title, tool) in enumerate(steps, 1):
        parts.append(f"""
<div class="iq-step">
  <div class="iq-step-num">{i}</div>
  <div class="iq-step-text">
    <div class="iq-step-title">{escape(title)}</div>
    <div class="iq-step-tool">{escape(tool)}</div>
  </div>
</div>""")
    return f'<div class="iq-stepper">{"".join(parts)}</div>'


# ── Video / PDF rows ─────────────────────────────────────────────────────────
def thumb(yt_id: str | None, icon_name: str = "movie") -> str:
    if yt_id:
        return (
            f'<div class="iq-thumb"><img src="https://img.youtube.com/vi/{escape(yt_id)}/mqdefault.jpg" '
            f'alt="" loading="lazy"></div>'
        )
    return f'<div class="iq-thumb iq-thumb-empty">{icon(icon_name, 26)}</div>'


def item_meta(title: str, subtitle: str, badges_html: str) -> str:
    sub = f'<div class="iq-item-sub">{escape(subtitle)}</div>' if subtitle else ""
    return f"""
<div class="iq-item-meta">
  <div class="iq-item-title">{escape(title)}</div>
  {sub}
  <div class="iq-badges">{badges_html}</div>
</div>"""


def empty_state(text: str, sub: str, icon_name: str = "movie") -> str:
    return f"""
<div class="iq-card iq-empty">
  <div class="iq-empty-icon">{icon(icon_name, 34)}</div>
  <div class="iq-empty-title">{escape(text)}</div>
  <div class="iq-empty-sub">{escape(sub)}</div>
</div>"""


# ── Answer cards (video Q&A) ─────────────────────────────────────────────────
def answer_card(timestamp: str, topic: str, summary: str, link: str = "") -> str:
    ts = ""
    if timestamp:
        inner = f'{icon("pin", 13)}{escape(timestamp)}'
        if link:
            inner += icon("play", 11)
            ts = f'<a class="iq-ts" href="{escape(link)}" target="_blank" rel="noopener">{inner}</a>'
        else:
            ts = f'<span class="iq-ts">{inner}</span>'
    topic_html = f'<div class="iq-answer-topic">{escape(topic)}</div>' if topic else ""
    return f"""
<div class="iq-card iq-answer">
  {ts}
  {topic_html}
  <div class="iq-answer-summary">{escape(summary)}</div>
</div>"""


# ── Pipeline phase card (Process Video) ──────────────────────────────────────
def phase_card(phase: str, label: str, state: str, icon_name: str) -> str:
    """state: wait | active | done"""
    status = {"wait": "Waiting", "active": "Running", "done": "Done"}[state]
    ic = icon("check", 18) if state == "done" else icon(icon_name, 18)
    return f"""
<div class="iq-card iq-phase iq-phase-{state}">
  <div class="iq-phase-icon">{ic}</div>
  <div class="iq-phase-name">{escape(phase)} · {status}</div>
  <div class="iq-phase-label">{escape(label)}</div>
</div>"""


def info_panel(sections: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
    """sections: list of (icon, heading, rows[(left, right)])."""
    blocks = []
    for icon_name, heading, rows in sections:
        rows_html = "".join(
            f'<div class="iq-info-row"><span>{escape(l)}</span><span>{escape(r)}</span></div>'
            for l, r in rows
        )
        blocks.append(
            f'<div class="iq-info-block"><div class="iq-info-heading">{icon(icon_name, 16)}'
            f'{escape(heading)}</div>{rows_html}</div>'
        )
    return f'<div class="iq-card iq-info">{"".join(blocks)}</div>'
