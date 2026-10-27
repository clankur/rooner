# Plotly Dark Theme

Standard colors and layout for all report charts. Matches the Quarto dark theme defined in `docs/theme-dark.scss`.

## Color palette

```python
# Page / chart backgrounds
PAPER_BG = "#1a1a2e"
PLOT_BG = "#16213e"
GRID_COLOR = "#2a2a4a"
FONT_COLOR = "#e0e0e0"

# Metric colors — extend as needed, keep distinguishable on dark bg
colors = {
    "reward": "#6ea8fe",
    "xp": "#f0ad4e",
    "actions": "#5cb85c",
    "valid": "#d9534f",
    "loss": "#c792ea",
    "kl": "#ff7979",
}
```

## Layout boilerplate

Apply to every figure:

```python
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(color="#e0e0e0"),
    height=600,  # adjust per chart; 350 for single-row
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.15,   # push further negative (-0.25) for short charts
        xanchor="center",
        x=0.5,
    ),
    title=dict(text="descriptive title", font=dict(size=16)),
)
```

## Grid styling

Apply to every axis:

```python
for i in range(1, rows + 1):
    for j in range(1, cols + 1):
        fig.update_xaxes(title_text="step", row=i, col=j, gridcolor="#2a2a4a")
        fig.update_yaxes(gridcolor="#2a2a4a", row=i, col=j)
```

## Subplot setup

```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Metric A", "Metric B", "Metric C", "Metric D"),
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
)
```

## Trace defaults

```python
fig.add_trace(go.Scatter(
    x=steps, y=values,
    mode="lines+markers",
    name="metric_name",
    line=dict(color=colors["metric_name"], width=3),
    marker=dict(size=8),
), row=1, col=1)
```

## Overlay bars

For comparing quantities (e.g., generated vs useful tokens):

```python
fig.add_trace(go.Bar(
    x=labels, y=total, name="total",
    marker_color="#6ea8fe", opacity=0.4,
))
fig.add_trace(go.Bar(
    x=labels, y=useful, name="useful",
    marker_color="#5cb85c",
))
fig.update_layout(barmode="overlay")
```
