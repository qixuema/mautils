from typing import Sequence, Optional, Union, Tuple
from collections import Counter
import warnings
import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_distribution(
    data: Sequence[Union[int, float]],
    *,
    discrete: bool = True,
    bins: Optional[Union[int, Sequence[Union[int, float]]]] = None,
    relative: bool = False,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    show_counts: bool = False,
    show_grid: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot value frequencies as a bar chart or histogram.

    Parameters
    ----------
    data : sequence of numbers
        Input values.
    discrete : bool, default True
        If True, plot each unique value; auto-switch if data not all ints.
    bins : int or sequence, optional
        Number of bins or bin edges for histogram mode.
    relative : bool, default False
        Plot relative frequencies instead of counts.
    figsize : tuple, default (8, 6)
        Figure size.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    show_counts : bool, default False
        Display counts or frequencies above bars.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object.
    """
    arr = np.asarray(data)

    # Switch to histogram if non-integers in discrete mode
    if discrete and not np.all(np.mod(arr, 1) == 0):
        warnings.warn("Non-integer values detected; switching to histogram.", UserWarning)
        discrete = False

    fig, ax = plt.subplots(figsize=figsize)

    if discrete:
        counts = Counter(arr.tolist())
        counts = Counter(arr.tolist())
        values, freqs = zip(*sorted(counts.items()))
        if relative:
            total = sum(freqs)
            freqs = [f / total for f in freqs]
            ylabel = "Relative Frequency"
        ax.bar(values, freqs)
        if show_counts:
            for x, y in zip(values, freqs):
                label = f"{y:.3f}" if relative else str(int(y))
                ax.text(x, y, label, ha='center', va='bottom')
    else:
        hist_kwargs = {"density": relative}
        if bins is not None:
            hist_kwargs["bins"] = bins
        n, bins_out, patches = ax.hist(arr, **hist_kwargs)
        if show_counts:
            for count, left, right in zip(n, bins_out[:-1], bins_out[1:]):
                label = f"{count:.3f}" if relative else str(int(count))
                ax.text((left+right)/2, count, label, ha='center', va='bottom')

    ax.set_title(title or ("Frequency Distribution" if not relative else "Relative Frequency Distribution"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return ax
