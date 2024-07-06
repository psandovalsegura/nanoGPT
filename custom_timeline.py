import torch
from torch.profiler._memory_profiler import Category, _CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX, MemoryProfileTimeline

def custom_memory_timeline(
    profile, path, device_str, figsize=(12, 8), title=None, ignore_categories=None,
) -> None:
    """Exports the memory timeline as an HTML file which contains
    the memory timeline plot embedded as a PNG file."""
    # Check if user has matplotlib installed, return gracefully if not.
    import importlib.util

    matplotlib_spec = importlib.util.find_spec("matplotlib")
    if matplotlib_spec is None:
        print(
            "export_memory_timeline_html failed because matplotlib was not found."
        )
        return

    import matplotlib.pyplot as plt
    import numpy as np

    mem_tl = MemoryProfileTimeline(profile._memory_profile())
    mt = mem_tl._coalesce_timeline(device_str)
    times, sizes = np.array(mt[0]), np.array(mt[1])
    # For this timeline, start at 0 to match Chrome traces.
    t_min = min(times)
    times -= t_min
    stacked = np.cumsum(sizes, axis=1) / 1024**3
    device = torch.device(device_str)
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)

    # Plot memory timeline as stacked data
    fig = plt.figure(figsize=figsize, dpi=80)
    axes = fig.gca()
    ignore_categories = [eval(i) for i in ignore_categories] if ignore_categories else []
    plot_categories = {
        category: color
        for category, color in _CATEGORY_TO_COLORS.items()
        if category not in ignore_categories
    }
    for category, color in plot_categories.items():
        i = _CATEGORY_TO_INDEX[category]
        axes.fill_between(
            times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
        )
    # Place legend at top right of inside of plot
    axes.legend(["Unknown" if i is None else i.name for i in plot_categories], loc="upper right")
    # Usually training steps are in magnitude of ms.
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Memory (GB)")
    title = "\n\n".join(
        ([title] if title else [])
        + [
            f"Max memory allocated: {max_memory_allocated/(1024**3):.2f} GiB \n"
            f"Max memory reserved: {max_memory_reserved/(1024**3):.2f} GiB"
        ]
    )
    axes.set_title(title)
    fig.savefig(f"{path}.png", format="png")
