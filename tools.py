# pyright: basic

# NOTE: This code is based on the code in the following articles:
# - https://raposa.trade/blog/how-to-think-about-cost-effective-risk-reduction-to-protect-your-portfolio/
# - https://raposa.trade/blog/how-to-gamble-with-demons-and-make-money-doing-it/
# - https://raposa.trade/blog/how-to-find-your-own-safe-haven-investing-strategy/
# Used with permission by Christian of Raposa Technologies, LLC.

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

PathsOrPath = npt.NDArray[np.float64]

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
BLUE = COLORS[0]
ORANGE = COLORS[1]
GREEN = COLORS[2]
RED = COLORS[3]
PURPLE = COLORS[4]


def quantile_path(
    trajectories: PathsOrPath, q: float = 0.5
) -> tuple[np.float64, PathsOrPath]:
    # The last column is the ending wealth of each path.
    ending_wealths = trajectories[:, -1]
    qth_ending_wealth = np.quantile(ending_wealths, q=q)
    # Select the path that ends closest to the qth quantile by subtracting the value
    # from all ending wealths, taking the absolute to make negative numbers go away, and
    # then use argmin to locate the row index where the ending wealth is zero (because
    # it was subtracted completely). That index is then the index of the desired row
    # (path) in trajectories.
    # https://stackoverflow.com/questions/53509154/how-can-i-find-the-indices-of-the-quantiles-of-an-array.
    path = trajectories[np.abs(qth_ending_wealth - ending_wealths).argmin()]
    return qth_ending_wealth, path


def geometric_mean(returns: Sequence[float] | PathsOrPath) -> float:
    return np.exp(np.log(returns).mean())


def draw_plot(
    trajectories: PathsOrPath,
    title: str,
    ylim_paths: tuple[int, int] | None = None,
    ylim_cagr: tuple[int, int] | None = None,
) -> None:
    _, path50 = quantile_path(trajectories)
    _, path95 = quantile_path(trajectories, q=0.95)
    _, path5 = quantile_path(trajectories, q=0.05)
    path_avg = trajectories.mean(axis=0)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(3, 1))
    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # ax.axhline(y=1, label="Break Even", color="k", linestyle=":")
    ax.plot(path95, label=r"$95^{th}$ Percentile", color=ORANGE)
    ax.plot(path50, label="Median", color=BLUE)
    ax.plot(path5, label=r"$5^{th}$ Percentile", color=GREEN)
    ax.plot(path_avg, label="Mean", linestyle=":", color=RED)
    ax.fill_between(
        np.arange(trajectories.shape[1]),
        y1=trajectories.min(axis=0),
        y2=trajectories.max(axis=0),
        alpha=0.3,
        color=PURPLE,
    )
    ax.set_title(title)
    ax.set_xlabel("Rolls")
    ax.set_ylabel("Ending Wealth")
    ax.semilogy()
    ax.legend(loc=3)

    # For the geometric growth, the last value of each trajectory is the ending wealth
    # of that trajectory (or EV in the CAGR formula, with BV being 1) or the product of
    # all values. The geometric growth is the nth root of the product of all values in
    # the trajectory, minus 1 to get at the percentage change, times 100 to give a nice
    # percentage value.
    n = 1 / trajectories.shape[1]  # Length of a row == number of rolls.
    growth = (np.power(trajectories[:, -1], n) - 1) * 100
    growth_avg = (np.power(path_avg[-1], n) - 1) * 100
    growth_perc5 = (np.power(path5[-1], n) - 1) * 100
    growth_perc5_outcome = (path5[-1] - 1) * 100
    growth_med = (np.power(path50[-1], n) - 1) * 100
    growth_med_outcome = (path50[-1] - 1) * 100
    growth_perc95 = (np.power(path95[-1], n) - 1) * 100
    growth_perc95_outcome = (path95[-1] - 1) * 100
    print(f"Average Growth Rate: {growth_avg:.2f}%")
    print(
        f"95th Percentile Growth Rate: {growth_perc95:.2f}% (outcome: {growth_perc95_outcome:.0f}%)"
    )
    print(f"Median Growth Rate: {growth_med:.2f}% (outcome: {growth_med_outcome:.0f}%)")
    print(
        f"5th Percentile Growth Rate: {growth_perc5:.2f}% (outcome: {growth_perc5_outcome:.0f}%)"
    )

    ax_hist.hist(growth, orientation="horizontal", bins=50, color=PURPLE, alpha=0.3)
    ax_hist.axhline(0, label="Break Even", color="k", linestyle=":")
    ax_hist.axhline(growth_perc95, label=r"$95^{th}$ Percentile", color=ORANGE)
    ax_hist.axhline(growth_med, label="Median", color=BLUE)
    ax_hist.axhline(growth_perc5, label=r"$5^{th}$ Percentile", color=GREEN)
    ax_hist.axhline(growth_avg, label="Mean", color=RED, linestyle=":")
    ax_hist.set_ylabel("Compound Growth Rate (%)")
    ax_hist.set_xlabel("Frequency")
    ax_hist.legend()

    if ylim_paths is not None:
        ax.set_ylim(*ylim_paths)
    if ylim_cagr is not None:
        ax_hist.set_ylim(*ylim_cagr)

    plt.tight_layout()
    plt.show()
