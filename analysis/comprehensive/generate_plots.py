#!/usr/bin/env python3
"""
Generate publication-quality figures for Agent Behavior Analysis report.
Refined academic aesthetic with careful typography and color choices.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import rcParams

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# Color palette: Elegant, cohesive palette inspired by scientific visualization
# Three-stratum colors used consistently across ALL figures
COLORS = {
    # Primary stratum colors (used consistently everywhere)
    'correct': '#3A7D44',      # Sage green - success/correct
    'wrong': '#E63946',        # Coral red - error/wrong
    'incomplete': '#457B9D',   # Steel blue - incomplete/in-progress
    
    # UI colors
    'bg': '#FAFAFA',           # Off-white background
    'text': '#2B2D42',         # Dark slate text
    'grid': '#E5E5E5',         # Subtle grid
}

# For gradient effects within single-color charts
def get_gradient_colors(base_hex, n, lightness_range=(0.35, 1.0)):
    """Generate gradient from light to dark based on a base color."""
    import colorsys
    # Convert hex to RGB then to HLS
    base_rgb = tuple(int(base_hex.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    
    colors = []
    for i, intensity in enumerate(np.linspace(lightness_range[0], lightness_range[1], n)):
        # Adjust lightness while keeping hue and saturation
        new_l = l * intensity + (1 - intensity) * 0.92
        new_s = s * (0.6 + 0.4 * intensity)  # Slightly desaturate lighter colors
        r, g, b = colorsys.hls_to_rgb(h, new_l, new_s)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return colors

# Typography settings - Times New Roman with exact specifications
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',  # Math font compatible with Times
    'font.size': 10,              # Main text: 10pt
    'axes.titlesize': 12,         # Titles: 12pt
    'axes.titleweight': 'bold',   # Titles: bold
    'axes.labelsize': 10,         # Axis labels: 10pt
    'axes.labelweight': 'normal', # Axis labels: normal weight
    'xtick.labelsize': 9,         # Tick labels: 9pt
    'ytick.labelsize': 9,         # Tick labels: 9pt
    'legend.fontsize': 9,         # Legend: 9pt
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})


def style_axis(ax, xlabel=None, ylabel=None):
    """Apply consistent styling to axis (no titles - use LaTeX captions)."""
    ax.set_facecolor(COLORS['bg'])
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS['text'])
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'], length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_color(COLORS['text'])
        spine.set_alpha(0.7)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Outcome Strata Distribution (Donut Chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_strata_distribution():
    """Create an elegant donut chart showing outcome distribution."""
    df = pd.read_csv('aba_stratum_counts.csv')
    
    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor='white')
    
    labels = ['Correct', 'Wrong', 'Incomplete']
    sizes = df['count'].values
    colors = [COLORS['correct'], COLORS['wrong'], COLORS['incomplete']]
    explode = (0.02, 0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,
        colors=colors,
        autopct='',
        startangle=90,
        explode=explode,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white', 'width': 0.65},
        pctdistance=0.75
    )
    
    # Add custom annotations with count and percentage
    total = sum(sizes)
    for i, (wedge, size) in enumerate(zip(wedges, sizes)):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        
        # Position for label outside
        ax.annotate(
            f'{labels[i]}\n{size:,} ({size/total*100:.1f}%)',
            xy=(x * 0.75, y * 0.75),
            ha='center', va='center',
            fontsize=9,
            fontweight='medium',
            color='white' if i != 2 else COLORS['text']
        )
    
    # Center annotation
    ax.text(0, 0, f'n = {total:,}', ha='center', va='center', 
            fontsize=14, fontweight='bold', color=COLORS['text'])
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['correct'], edgecolor='white', label='Complete-Correct'),
        mpatches.Patch(facecolor=COLORS['wrong'], edgecolor='white', label='Complete-Wrong'),
        mpatches.Patch(facecolor=COLORS['incomplete'], edgecolor='white', label='Incomplete'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/aba_strata_distribution.pdf', facecolor='white')
    plt.close()
    print("✓ Saved: aba_strata_distribution.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Outcomes by Message Limit (Stacked Bar with annotations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_outcomes_by_limit():
    """Create refined stacked bar chart for budget analysis."""
    # Data from the paper text
    limits = [15, 30, 50]
    correct = [29.2, 37.9, 39.9]
    wrong = [25.7, 36.1, 44.2]
    incomplete = [45.1, 26.0, 15.9]
    
    fig, ax = plt.subplots(figsize=(5, 3.8), facecolor='white')
    ax.set_facecolor(COLORS['bg'])
    
    x = np.arange(len(limits))
    width = 0.55
    
    # Stacked bars
    bars1 = ax.bar(x, correct, width, label='Correct', color=COLORS['correct'],
                   edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, wrong, width, bottom=correct, label='Wrong', color=COLORS['wrong'],
                   edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x, incomplete, width, bottom=np.array(correct)+np.array(wrong),
                   label='Incomplete', color=COLORS['incomplete'], edgecolor='white', linewidth=1.5)
    
    # Add percentage labels inside bars
    def add_bar_labels(bars, values, bottoms):
        for bar, val, bottom in zip(bars, values, bottoms):
            height = bar.get_height()
            if height > 8:  # Only label if segment is big enough
                ax.text(bar.get_x() + bar.get_width()/2, bottom + height/2,
                       f'{val:.1f}%', ha='center', va='center', 
                       fontsize=8, fontweight='medium', color='white')
    
    add_bar_labels(bars1, correct, [0]*3)
    add_bar_labels(bars2, wrong, correct)
    add_bar_labels(bars3, incomplete, np.array(correct)+np.array(wrong))
    
    ax.set_xlabel('Message Limit', color=COLORS['text'])
    ax.set_ylabel('Proportion of Runs (%)', color=COLORS['text'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(limits, fontweight='medium')
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    
    # Legend at top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, 
              frameon=False, fontsize=9, columnspacing=1.5)
    
    # Add subtle gridlines
    ax.yaxis.grid(True, color=COLORS['grid'], linestyle='-', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Style spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['text'])
        ax.spines[spine].set_alpha(0.7)
    ax.tick_params(colors=COLORS['text'], length=4, width=0.8)
    
    plt.tight_layout()
    plt.savefig('figures/aba_outcomes_by_message_limit.pdf', facecolor='white')
    plt.close()
    print("✓ Saved: aba_outcomes_by_message_limit.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Failure Taxonomy (Horizontal Bar with value labels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_failure_taxonomy():
    """Create refined horizontal bar chart for failure taxonomy."""
    df = pd.read_csv('aba_failure_taxonomy.csv')
    df = df.sort_values('count', ascending=True)
    
    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor='white')
    ax.set_facecolor(COLORS['bg'])
    
    y = np.arange(len(df))
    
    # Use 'wrong' color gradient (coral red) since this is about failures
    colors = get_gradient_colors(COLORS['wrong'], len(df), (0.4, 1.0))
    
    bars = ax.barh(y, df['count'], height=0.7, color=colors, 
                   edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2,
               f'{int(row["count"])} ({row["percent"]:.1f}%)',
               va='center', ha='left', fontsize=8, color=COLORS['text'])
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['failure_type'], fontsize=9)
    ax.set_xlim(0, max(df['count']) * 1.35)
    
    style_axis(ax, xlabel='Count')
    
    # Remove y-axis spine for cleaner look
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    
    ax.xaxis.grid(True, color=COLORS['grid'], linestyle='-', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('figures/aba_failure_taxonomy_wrong.pdf', facecolor='white')
    plt.close()
    print("✓ Saved: aba_failure_taxonomy_wrong.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Non-convergence Categories (Horizontal Bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_nonconvergence():
    """Create refined horizontal bar chart for non-convergence categories."""
    df = pd.read_csv('aba_nonconvergence.csv')
    df = df.sort_values('count', ascending=True)
    
    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='white')
    ax.set_facecolor(COLORS['bg'])
    
    y = np.arange(len(df))
    
    # Use 'incomplete' color gradient (steel blue) since this is about incomplete runs
    colors = get_gradient_colors(COLORS['incomplete'], len(df), (0.35, 1.0))
    
    bars = ax.barh(y, df['count'], height=0.7, color=colors,
                   edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + 3, bar.get_y() + bar.get_height()/2,
               f'{int(row["count"])} ({row["percent"]:.1f}%)',
               va='center', ha='left', fontsize=8, color=COLORS['text'])
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['nonconv_type'], fontsize=9)
    ax.set_xlim(0, max(df['count']) * 1.35)
    
    style_axis(ax, xlabel='Count')
    
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    
    ax.xaxis.grid(True, color=COLORS['grid'], linestyle='-', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('figures/aba_nonconvergence_categories.pdf', facecolor='white')
    plt.close()
    print("✓ Saved: aba_nonconvergence_categories.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Bottleneck Categories (Horizontal Bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bottlenecks():
    """Create refined horizontal bar chart for bottleneck categories."""
    df = pd.read_csv('aba_bottlenecks.csv')
    df = df.sort_values('count', ascending=True)
    
    fig, ax = plt.subplots(figsize=(5.5, 4), facecolor='white')
    ax.set_facecolor(COLORS['bg'])
    
    y = np.arange(len(df))
    
    # Use a blend/neutral color since this covers both wrong + incomplete
    # Using a muted purple-gray that complements both red and blue
    blend_color = '#6C5B7B'  # Muted purple
    colors = get_gradient_colors(blend_color, len(df), (0.35, 1.0))
    
    bars = ax.barh(y, df['count'], height=0.7, color=colors,
                   edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + 8, bar.get_y() + bar.get_height()/2,
               f'{int(row["count"])} ({row["percent"]:.1f}%)',
               va='center', ha='left', fontsize=8, color=COLORS['text'])
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['bottleneck'], fontsize=9)
    ax.set_xlim(0, max(df['count']) * 1.25)
    
    style_axis(ax, xlabel='Count')
    
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    
    ax.xaxis.grid(True, color=COLORS['grid'], linestyle='-', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('figures/aba_bottleneck_categories.pdf', facecolor='white')
    plt.close()
    print("✓ Saved: aba_bottleneck_categories.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Correct Run Flags (Horizontal Bar with fractions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_correct_flags():
    """Create refined horizontal bar chart for judge flags in correct runs."""
    # Data derived from the plot (fraction of correct runs)
    data = {
        'flag': ['Verification noted', 'Unsupported claims noted', 
                 'Lucky/guessing noted', 'Redundancy noted'],
        'fraction': [0.16, 0.27, 0.43, 0.79]
    }
    df = pd.DataFrame(data)
    df = df.sort_values('fraction', ascending=True)
    
    fig, ax = plt.subplots(figsize=(5.5, 3), facecolor='white')
    ax.set_facecolor(COLORS['bg'])
    
    y = np.arange(len(df))
    
    # Use 'correct' color gradient (sage green) since this is about correct runs
    colors = get_gradient_colors(COLORS['correct'], len(df), (0.45, 1.0))
    
    bars = ax.barh(y, df['fraction'], height=0.65, color=colors,
                   edgecolor='white', linewidth=1)
    
    # Add percentage labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{row["fraction"]*100:.0f}%',
               va='center', ha='left', fontsize=9, fontweight='medium', 
               color=COLORS['text'])
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['flag'], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    style_axis(ax, xlabel='Fraction of Correct Runs')
    
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    
    ax.xaxis.grid(True, color=COLORS['grid'], linestyle='-', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('figures/aba_correct_run_flags.pdf', facecolor='white')
    plt.close()
    print("✓ Saved: aba_correct_run_flags.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Publication-Quality Figures")
    print("="*60 + "\n")
    
    plot_strata_distribution()
    plot_outcomes_by_limit()
    plot_failure_taxonomy()
    plot_nonconvergence()
    plot_bottlenecks()
    plot_correct_flags()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60 + "\n")
