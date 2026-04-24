# CostSherlock Dashboard — Design System Notes

*Priti Ghosh — Northeastern University — INFO 7375 Generative AI*

---

## Design Philosophy

The CostSherlock dashboard is styled to feel like an **internal engineering tool** —
not a consumer product. The visual language borrows from AWS Console and developer
tools: high information density, muted colors, and reserved use of color to signal
severity rather than decoration.

Key principle: **color carries meaning, not style.**

---

## Color Palette

### Primary Backgrounds

| Name | Hex | Usage |
|------|-----|-------|
| Navy | `#0F1E38` | Page background, header bar, sidebar |
| Card | `#1B2A4A` | Metric cards, investigation containers |
| Dark Navy | `#0A1628` | Deep shadows, hover states |

### Accent Colors (semantic)

| Name | Hex | Tailwind Equiv | Usage |
|------|-----|----------------|-------|
| Blue | `#2563EB` | Blue-600 | Primary interactive, borders, links |
| Emerald | `#059669` | Emerald-600 | Success states, "PASS" badges, investigated buttons |
| Amber | `#D97706` | Amber-600 | Warning states, medium confidence |
| Red | `#DC2626` | Red-600 | Error states, critical severity |
| Light Blue | `#93C5FD` | Blue-300 | Sidebar headers, secondary labels |

### Text Hierarchy

| Level | Hex | Usage |
|-------|-----|-------|
| Primary | `#FFFFFF` | Headlines, key values, metric numbers |
| Secondary | `#CBD5E1` | Body text, descriptions |
| Muted | `#94A3B8` | Captions, timestamps, secondary labels |
| Disabled | `#64748B` | Placeholder text, inactive states |

---

## Typography

Uses Streamlit's default system font stack (Inter / San Francisco / Segoe UI).
No custom font imports — keeps the dashboard fast and self-contained.

Font size conventions:
- Metric card values: `2rem`, bold
- Section headers: `1.25rem`, semibold
- Body text: `1rem`, regular
- Captions / labels: `0.85rem`, regular
- Code / event names: monospace, `0.9rem`

---

## Severity System

Anomalies and hypotheses use a three-tier severity system mapped to z-scores:

| Tier | Z-score | Color | Badge class |
|------|---------|-------|-------------|
| Critical | z ≥ 4.0 | Red `#DC2626` | `.sev-critical` |
| Warning | 3.0 ≤ z < 4.0 | Amber `#D97706` | `.sev-warning` |
| Info | 2.5 ≤ z < 3.0 | Blue `#2563EB` | `.sev-info` |

---

## Confidence Color Coding

Analyst confidence scores (0.0–1.0) are mapped to colors in the Investigation view:

| Range | Color | Label |
|-------|-------|-------|
| ≥ 0.75 | Green `#059669` | High |
| 0.50–0.74 | Amber `#D97706` | Medium |
| < 0.50 | Red `#DC2626` | Low |

---

## Component Patterns

### Metric Cards
- Background: translucent blue overlay on card base (`rgba(37, 99, 235, 0.07)`)
- Text color: inherits from `var(--text-color)` for theme compatibility
- Border: `1px solid rgba(37, 99, 235, 0.2)`
- No fixed heights — content-driven

### Status Banners
- Full-width horizontal bar below navigation
- Color fills the left border (4px) to signal severity
- Background tinted at 8–12% opacity of the severity color

### Navigation
- Sidebar radio group bound to `st.session_state.current_view` via `key=` parameter
- All view transitions use `on_click` callbacks to avoid rerun race conditions
- No `st.rerun()` calls — state changes trigger natural Streamlit reruns

---

## Accessibility Notes

- All color combinations achieve at minimum **WCAG AA** contrast ratio (4.5:1) for body text
- Interactive elements include hover states visible without color (border changes)
- Severity indicators use both color **and** text labels (never color alone)
- The loading shimmer animation respects `prefers-reduced-motion` via CSS `@media` query

---

## Streamlit Version Compatibility

Tested on Streamlit **1.x**. Key internal selectors used:

| Selector | Purpose |
|----------|---------|
| `[data-testid="stHeader"]` | Top navigation bar |
| `[data-testid="stSidebar"]` | Left sidebar |
| `[data-testid="stMetric"]` | Metric card containers |
| `[data-testid="stMarkdownContainer"]` | Report markdown blocks |

These selectors are stable within Streamlit 1.x but should be reviewed on major upgrades.
