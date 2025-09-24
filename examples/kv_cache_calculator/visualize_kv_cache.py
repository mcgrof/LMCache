#!/usr/bin/env python3
"""
KV Cache Size Visualization for Large Context Lengths
Generates static PNG charts comparing memory requirements across models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import os

# Import shared utilities
from kv_cache_utils import (
    load_model_configs,
    calculate_kv_cache_size,
    select_representative_models,
    format_yaxis_memory,
    get_model_color,
    GPU_CONSUMER_MEMORY,
    GPU_H100_MEMORY,
    GPU_MULTI_160_MEMORY,
    GPU_MULTI_320_MEMORY,
    GPU_MULTI_640_MEMORY,
    FEASIBILITY_COLORS,
)




def create_comparison_chart(configs, dtype="float16"):
    """Create a comprehensive comparison chart for multiple context lengths"""
    models = select_representative_models(configs)

    # Define context lengths to visualize (log scale from 1K to 1M+)
    context_lengths = [
        1_000,  # 1K
        2_000,  # 2K
        4_000,  # 4K
        8_000,  # 8K
        16_000,  # 16K
        32_000,  # 32K
        64_000,  # 64K
        128_000,  # 128K
        256_000,  # 256K
        512_000,  # 512K
        1_000_000,  # 1M
        1_500_000,  # 1.5M
    ]

    # Calculate cache sizes for each model and context length
    cache_sizes = {}
    for model_name, config in models.items():
        cache_sizes[model_name] = [
            calculate_kv_cache_size(config, ctx_len, dtype, model_name)
            for ctx_len in context_lengths
        ]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # Plot 1: Log-scale line chart
    ax1 = fig.add_subplot(gs[0, :])
    for model_name, sizes in cache_sizes.items():
        short_name = model_name.split("/")[-1]
        color = get_model_color(model_name)
        ax1.loglog(
            context_lengths,
            sizes,
            marker="o",
            label=short_name,
            linewidth=2,
            markersize=6,
            color=color,
        )

    ax1.set_xlabel("Context Length (tokens)", fontsize=12)
    ax1.set_ylabel("KV Cache Size", fontsize=12)
    ax1.set_title(
        f"KV Cache Memory Requirements vs Context Length ({dtype})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc="upper left", fontsize=9, ncol=2)

    # Format y-axis with human-readable units
    ax1.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Add context length labels
    ax1.set_xticks(context_lengths)
    ax1.set_xticklabels(
        [
            "1K",
            "2K",
            "4K",
            "8K",
            "16K",
            "32K",
            "64K",
            "128K",
            "256K",
            "512K",
            "1M",
            "1.5M",
        ],
        rotation=45,
    )

    # Plot 2: Bar chart for specific context lengths
    ax2 = fig.add_subplot(gs[1, 0])
    selected_contexts = [8_000, 128_000, 1_000_000]
    x_pos = np.arange(len(models))
    width = 0.25

    for i, ctx_len in enumerate(selected_contexts):
        values = [
            calculate_kv_cache_size(config, ctx_len, dtype, name)
            for name, config in models.items()
        ]
        colors = [get_model_color(name) for name in models.keys()]
        bars = ax2.bar(
            x_pos + i * width,
            values,
            width,
            label=f"{ctx_len//1000}K tokens",
            alpha=0.8,
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("KV Cache Size", fontsize=11)
    ax2.set_title(
        "Memory Requirements at Key Context Lengths", fontsize=12, fontweight="bold"
    )
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(
        [m.split("/")[-1] for m in models.keys()], rotation=45, ha="right", fontsize=8
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    # Format y-axis with human-readable units
    ax2.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Plot 3: Heatmap showing memory feasibility
    ax3 = fig.add_subplot(gs[1, 1])

    # Create feasibility matrix
    model_names = list(models.keys())
    short_model_names = [m.split("/")[-1] for m in model_names]
    contexts_for_heatmap = [
        8_000,
        16_000,
        32_000,
        64_000,
        128_000,
        256_000,
        512_000,
        1_000_000,
    ]

    # Calculate which combinations are feasible for different GPU sizes
    feasibility = np.zeros((len(model_names), len(contexts_for_heatmap)))
    for i, (model_name, config) in enumerate(models.items()):
        for j, ctx_len in enumerate(contexts_for_heatmap):
            size = calculate_kv_cache_size(config, ctx_len, dtype, model_name)
            if size <= GPU_CONSUMER_MEMORY:
                feasibility[i, j] = 1  # Green - fits in consumer GPU
            elif size <= GPU_H100_MEMORY:
                feasibility[i, j] = 2  # Yellow - needs datacenter GPU
            elif size <= GPU_MULTI_320_MEMORY:
                feasibility[i, j] = 3  # Orange - needs multiple GPUs
            else:
                feasibility[i, j] = 4  # Red - very challenging

    # Create custom colormap
    colors_list = list(FEASIBILITY_COLORS.values())
    cmap = ListedColormap(colors_list)

    im = ax3.imshow(feasibility, cmap=cmap, aspect="auto", vmin=1, vmax=4)
    ax3.set_xticks(range(len(contexts_for_heatmap)))
    ax3.set_xticklabels([f"{c//1000}K" for c in contexts_for_heatmap], fontsize=9)
    ax3.set_yticks(range(len(short_model_names)))
    ax3.set_yticklabels(short_model_names, fontsize=8)
    ax3.set_xlabel("Context Length", fontsize=11)
    ax3.set_ylabel("Model", fontsize=11)
    ax3.set_title("GPU Memory Feasibility Matrix", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(contexts_for_heatmap)):
            size = calculate_kv_cache_size(
                models[model_names[i]], contexts_for_heatmap[j], dtype, model_names[i]
            )
            text_color = "white" if feasibility[i, j] >= 3 else "black"
            ax3.text(
                j,
                i,
                f"{size:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    # Add legend
    legend_elements = [
        mpatches.Patch(color=FEASIBILITY_COLORS['fits_consumer'],
                      label=f"≤{GPU_CONSUMER_MEMORY} GiB (Consumer)"),
        mpatches.Patch(color=FEASIBILITY_COLORS['fits_datacenter'],
                      label=f"≤{GPU_H100_MEMORY} GiB (A100/H100)"),
        mpatches.Patch(color=FEASIBILITY_COLORS['needs_multi'],
                      label=f"≤{GPU_MULTI_320_MEMORY} GiB (Multi-GPU)"),
        mpatches.Patch(color=FEASIBILITY_COLORS['challenging'],
                      label=f">{GPU_MULTI_320_MEMORY} GiB (Challenging)"),
    ]
    ax3.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8
    )

    # Add main title
    fig.suptitle(
        "KV Cache Memory Analysis for Large Language Models",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    return fig


def create_extreme_context_projection(configs, dtype="float16"):
    """Create visualization focusing on extreme context lengths (512K-2M)"""
    models = select_representative_models(configs)

    # Focus on very large contexts
    extreme_contexts = [
        512_000,  # 512K
        750_000,  # 750K
        1_000_000,  # 1M
        1_250_000,  # 1.25M
        1_500_000,  # 1.5M
        1_750_000,  # 1.75M
        2_000_000,  # 2M
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Calculate sizes
    cache_sizes = {}
    for model_name, config in models.items():
        cache_sizes[model_name] = [
            calculate_kv_cache_size(config, ctx_len, dtype, model_name)
            for ctx_len in extreme_contexts
        ]

    # Plot 1: Line chart for extreme contexts
    for model_name, sizes in cache_sizes.items():
        short_name = model_name.split("/")[-1]
        ax1.plot(
            extreme_contexts,
            sizes,
            marker="o",
            label=short_name,
            linewidth=2,
            markersize=8,
        )

    ax1.set_xlabel("Context Length (millions of tokens)", fontsize=12)
    ax1.set_ylabel("KV Cache Size", fontsize=12)
    ax1.set_title(
        "Memory Requirements for Extreme Context Lengths",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Format y-axis with human-readable units
    ax1.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Format x-axis
    ax1.set_xticks(extreme_contexts)
    ax1.set_xticklabels([f"{c/1_000_000:.2f}M" for c in extreme_contexts])

    # Add horizontal lines for GPU memory limits
    gpu_limits = [
        (GPU_H100_MEMORY, f"H100 {GPU_H100_MEMORY} GiB", "blue"),
        (GPU_MULTI_160_MEMORY, "2x H100", "green"),
        (GPU_MULTI_320_MEMORY, "4x H100", "orange"),
        (GPU_MULTI_640_MEMORY, "8x H100", "red"),
    ]

    for limit, label, color in gpu_limits:
        ax1.axhline(y=limit, color=color, linestyle="--", alpha=0.5, label=label)

    # Plot 2: Stacked bar showing memory breakdown at 1M context
    ctx_1m = 1_000_000
    model_names = list(models.keys())
    short_names = [m.split("/")[-1] for m in model_names]
    sizes_1m = [
        calculate_kv_cache_size(models[m], ctx_1m, dtype, m) for m in model_names
    ]

    # Sort by size
    sorted_indices = np.argsort(sizes_1m)
    sorted_names = [short_names[i] for i in sorted_indices]
    sorted_sizes = [sizes_1m[i] for i in sorted_indices]

    bars = ax2.barh(range(len(sorted_names)), sorted_sizes)

    # Color bars based on feasibility
    for i, (bar, size) in enumerate(zip(bars, sorted_sizes)):
        if size <= GPU_H100_MEMORY:
            bar.set_color(FEASIBILITY_COLORS['fits_consumer'])  # Green
        elif size <= GPU_MULTI_160_MEMORY:
            bar.set_color(FEASIBILITY_COLORS['fits_datacenter'])  # Yellow
        elif size <= GPU_MULTI_320_MEMORY:
            bar.set_color(FEASIBILITY_COLORS['needs_multi'])  # Orange
        else:
            bar.set_color(FEASIBILITY_COLORS['challenging'])  # Red

    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names, fontsize=9)
    ax2.set_xlabel("KV Cache Size", fontsize=12)
    ax2.set_title(
        f"Memory Requirements at 1M Tokens ({dtype})", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, axis="x", alpha=0.3)

    # Format x-axis with human-readable units
    ax2.xaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

    # Add vertical lines for GPU limits - position text to avoid overlap
    for i, (limit, label, color) in enumerate(gpu_limits[:3]):
        ax2.axvline(x=limit, color=color, linestyle="--", alpha=0.5)
        # Stagger text positions to avoid overlap
        y_pos = len(sorted_names) - 0.5 - (i * 0.7)
        ax2.text(
            limit + 5,
            y_pos,
            label,
            rotation=0,
            verticalalignment="center",
            fontsize=8,
            color=color,
        )

    # Add value labels
    for i, (name, size) in enumerate(zip(sorted_names, sorted_sizes)):
        ax2.text(size + 5, i, f"{size:.0f} GiB", va="center", fontsize=8)

    plt.suptitle(
        "Extreme Context Length Analysis (512K-2M tokens)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def create_dtype_comparison(configs):
    """Create comparison across different data types"""
    models = select_representative_models(configs)

    # Select a few key models for cleaner visualization
    key_models = {
        "meta-llama/Llama-3.1-8B-Instruct": models["meta-llama/Llama-3.1-8B-Instruct"],
        "meta-llama/Llama-3.1-70B-Instruct": models[
            "meta-llama/Llama-3.1-70B-Instruct"
        ],
        "deepseek-ai/DeepSeek-V3": models["deepseek-ai/DeepSeek-V3"],
    }

    dtypes = ["int8", "float16", "bfloat16", "float32"]
    context_lengths = [8_000, 32_000, 128_000, 512_000, 1_000_000]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (model_name, config) in enumerate(key_models.items()):
        ax = axes[idx]
        short_name = model_name.split("/")[-1]

        # Calculate sizes for each dtype and context
        for dtype in dtypes:
            sizes = [
                calculate_kv_cache_size(config, ctx, dtype, model_name)
                for ctx in context_lengths
            ]
            ax.semilogy(context_lengths, sizes, marker="o", label=dtype, linewidth=2)

        ax.set_xlabel("Context Length (tokens)", fontsize=11)
        ax.set_ylabel("KV Cache Size", fontsize=11)
        ax.set_title(f"{short_name}", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=9)

        # Format y-axis with human-readable units
        ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis_memory))

        # Format x-axis
        ax.set_xticks(context_lengths)
        ax.set_xticklabels(["8K", "32K", "128K", "512K", "1M"], rotation=45)

    plt.suptitle(
        "Impact of Data Type on KV Cache Memory Requirements",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig




def main():
    """Main function to generate all visualizations"""
    # Load configurations
    configs = load_model_configs()

    print("Generating KV Cache visualizations...")

    # Create output directory
    os.makedirs("kv_cache_visualizations", exist_ok=True)

    # Generate comprehensive comparison chart
    fig1 = create_comparison_chart(configs, dtype="float16")
    fig1.savefig(
        "kv_cache_visualizations/kv_cache_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Created: kv_cache_comparison.png")

    # Generate extreme context projection
    fig2 = create_extreme_context_projection(configs, dtype="float16")
    fig2.savefig(
        "kv_cache_visualizations/extreme_context_projection.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("✓ Created: extreme_context_projection.png")

    # Generate dtype comparison
    fig3 = create_dtype_comparison(configs)
    fig3.savefig(
        "kv_cache_visualizations/dtype_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Created: dtype_comparison.png")

    print(
        "\nAll visualizations generated successfully in 'kv_cache_visualizations/' directory"
    )
    print(
        "\nNote: Run generate_memory_table.py to create the detailed memory requirements table"
    )
    print("\nKey insights:")
    print("- Small models (1-3B) can handle 1M context with 80-160 GiB memory")
    print("- Medium models (7-8B) require 200-300 GiB for 1M context")
    print("- Large models (70B+) need multiple GPUs for contexts >128K")
    print("- DeepSeek-V3 uses KV-LoRA optimization for significant memory savings")
    print("- Using int8 quantization can reduce memory by 50% vs float16")


if __name__ == "__main__":
    main()
