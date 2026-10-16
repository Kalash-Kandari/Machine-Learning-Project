import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def huber_loss(error, delta=1.0):
    error = np.asarray(error)
    abs_error = np.abs(error)

    return np.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta)
    )


delta = 1.0

x = np.linspace(-4, 4, 500)

huber = huber_loss(x, delta)
squared = 0.5 * x**2
absolute = np.abs(x)

animated_errors = np.concatenate([
    np.linspace(-4, 4, 160),
    np.linspace(4, -4, 160)
])


fig, ax = plt.subplots(figsize=(10, 6))

fig.suptitle(
    "Huber Loss - Graphical Animation",
    fontsize=16,
    fontweight="bold"
)

ax.set_title(
    "Comparison with Squared Error and Absolute Error"
)

ax.set_xlabel("Residual  r = y - ŷ")
ax.set_ylabel("Loss")

ax.set_xlim(-4.2, 4.2)
ax.set_ylim(0, 8.5)

ax.grid(True, alpha=0.3)

ax.plot(
    x,
    squared,
    linewidth=2,
    linestyle="--",
    label="Squared Error"
)

ax.plot(
    x,
    absolute,
    linewidth=2,
    linestyle=":",
    label="Absolute Error"
)

ax.plot(
    x,
    huber,
    linewidth=3,
    label=f"Huber Loss (δ = {delta})"
)

ax.axvline(
    -delta,
    linestyle="--",
    alpha=0.6
)

ax.axvline(
    delta,
    linestyle="--",
    alpha=0.6
)

ax.text(
    -delta,
    7.7,
    "-δ",
    ha="center",
    fontsize=12
)

ax.text(
    delta,
    7.7,
    "+δ",
    ha="center",
    fontsize=12
)


ax.text(
    0,
    6.8,
    "QUADRATIC REGION\n|r| ≤ δ",
    ha="center",
    fontsize=11,
    bbox=dict(
        boxstyle="round",
        alpha=0.15
    )
)

ax.text(
    -2.7,
    6.0,
    "LINEAR\nREGION",
    ha="center",
    fontsize=11
)

ax.text(
    2.7,
    6.0,
    "LINEAR\nREGION",
    ha="center",
    fontsize=11
)


point, = ax.plot(
    [],
    [],
    marker="o",
    markersize=10
)

vertical_line, = ax.plot(
    [],
    [],
    linestyle="--",
    alpha=0.7
)

residual_text = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.15
    )
)

region_text = ax.text(
    0.02,
    0.80,
    "",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top"
)

formula_text = ax.text(
    0.98,
    0.95,
    "",
    transform=ax.transAxes,
    fontsize=11,
    horizontalalignment="right",
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.15
    )
)

ax.legend(loc="upper right")

def init():

    point.set_data([], [])
    vertical_line.set_data([], [])

    residual_text.set_text("")
    region_text.set_text("")
    formula_text.set_text("")

    return (
        point,
        vertical_line,
        residual_text,
        region_text,
        formula_text
    )

def update(frame):

    error = animated_errors[frame]

    loss = float(
        huber_loss(error, delta)
    )

    point.set_data(
        [error],
        [loss]
    )

    vertical_line.set_data(
        [error, error],
        [0, loss]
    )

    if abs(error) <= delta:

        region = "Quadratic Region"

        formula = (
            "Using quadratic formula:\n"
            "L = 1/2 r²"
        )

    else:

        region = "Linear Region"

        formula = (
            "Using linear formula:\n"
            "L = δ(|r| - 1/2 δ)"
        )

    residual_text.set_text(
        f"Residual r = {error:.2f}\n"
        f"Huber Loss = {loss:.2f}"
    )

    region_text.set_text(
        f"Current Region:\n{region}"
    )

    formula_text.set_text(formula)

    return (
        point,
        vertical_line,
        residual_text,
        region_text,
        formula_text
    )

animation = FuncAnimation(
    fig,
    update,
    frames=len(animated_errors),
    init_func=init,
    interval=40,
    blit=False,
    repeat=True
)

plt.tight_layout()
plt.show()