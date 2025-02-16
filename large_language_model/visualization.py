import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configurable Parameters
OUTPUT_DIR = "./fig_results"  # Directory to save results
DATA_OBJ_DIR = "objective_test/results"
DATA_DIR = "subjective_test/results"  # Directory containing input data
DATE = datetime.now().strftime('%y%m%d')  # Current date in YYMMDD format
TARGETS = ['Objective', 'Baseline', 'Representativeness Bias', 'Attribution Error', 'Group Bias']  # Targets to analyze

# Model and attribute visualization settings
MODEL_COLORS = {
    "gpt-3.5-turbo-0125": "blue",
    "gpt-4o-2024-08-06": "orange",
    "gemini-1.5-pro": "red",
    "Llama-3.2-90B-Vision-Instruct": "green",
    "WizardLM-2-8x22B": "purple",
    "Qwen2.5-72B-Instruct": "brown",
    "gpt-4o-mini": "brown"
}
ATTRIBUTE_MARKERS = {"race": "o", "gender": "^"}

# Ensure Output Directory Exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Function to Calculate Entropy
def calculate_entropy(a, k=2):
    return - (1 / np.log(k)) * (a * np.log(a) + (1 - a) * np.log((1 - a) / (k - 1)))


# Main Loop for Target Analysis
for target in TARGETS:
    # Construct file path for current target
    if target == 'Objective':
        file_path = f"{DATA_OBJ_DIR}/{DATE}/llm_{target.replace(' ', '_').lower()}_result.csv"
    else:
        file_path = f"{DATA_DIR}/{DATE}/llm_{target.replace(' ', '_').lower()}_result.csv"

    # Load data
    df = pd.read_csv(file_path)

    # Plot setup based on target
    if target in ['Objective', 'Baseline', 'Representativeness Bias']:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 11), gridspec_kw={'height_ratios': [7, 0.5]})

    if target != 'Objective':
        # Plot maximum entropy lines
        max_entropy_lines = []
        for k, color in zip([2, 4], ["blue", "red"]):
            a_values = np.linspace(0.5, 0.7, 500) if k == 2 else np.linspace(0.25, 0.6, 500)
            entropy_values = calculate_entropy(a_values, k)
            line, = ax.plot(100 * a_values, 100 * entropy_values, label=f'Max Entropy (k={k})', color=color)
            max_entropy_lines.append(line)

    # Scatter plot for model data
    for _, row in df.iterrows():
        model = row['Model']
        attribute = row['Attribute']
        entropy = row['Score_entropy'] if target == 'Objective' else row['Score_fair']
        accuracy = row['Score_fact']
        ax.scatter(
            accuracy, entropy,
            color=MODEL_COLORS[model],
            marker=ATTRIBUTE_MARKERS[attribute],
            s=500, edgecolor="black", alpha=0.5,
            label=model if attribute == 'race' else ""
        )

    # Add markers for attributes
    for attribute, marker in ATTRIBUTE_MARKERS.items():
        ax.scatter([], [], color='black', marker=marker, s=500, label=attribute.capitalize())

    # Set custom target name for the title
    target_name = 'In-group/Out-group Bias' if target == 'Group Bias' else target

    # Customize plot appearance
    ax.set_xlabel(r'$S_{\mathrm{fact}}$', fontsize=44)
    if target == 'Objective':
        ax.set_ylabel(r'$S_{fair}$', fontsize=44)
    else:
        ax.set_ylabel(r'$S_{\mathrm{E}}$', fontsize=44)
    ax.tick_params(axis='both', which='major', labelsize=34)
    ax.set_xlim(22, 72)
    ax.grid(color='gray', linestyle='--', linewidth=2, alpha=0.7)
    if target == 'Objective':
        ax.set_title(f'Objective Test Scores ({target_name})', fontsize=36, pad=20)
    else:
        ax.set_title(f'LLM Trade-offs ({target_name})', fontsize=36, pad=20)

    # Enhance axis aesthetics
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # Customize legend order
    handles, labels = ax.get_legend_handles_labels()
    custom_order = [
        'Max Entropy (k=2)', 'Max Entropy (k=4)', *MODEL_COLORS.keys(), 'Gender', 'Race'
    ]
    order = [labels.index(item) for item in custom_order if item in labels]
    sorted_handles = [handles[i] for i in order]
    sorted_labels = [labels[i] for i in order]
    ax.legend(sorted_handles, sorted_labels, loc='lower left', fontsize=20).get_frame().set_alpha(0.4)

    if target not in ['Objective', 'Baseline', 'Representativeness Bias']:
        # Bottom plot: Background influence scatter
        for i, row in df.iterrows():
            model = row["Model"]
            attribute = row["Attribute"]
            background_influence = row["Background"] / 100
            ax2.scatter(
                background_influence, 0,
                color=MODEL_COLORS[model],
                marker=ATTRIBUTE_MARKERS[attribute],
                s=400, edgecolor="black", alpha=0.5
            )

        # Add reference lines
        ax2.axvline(0.25, color='black', linestyle='-', linewidth=2, label='Reference 0.25')
        ax2.axvline(0.5, color='black', linestyle='-', linewidth=2, label='Reference 0.5')

        # Customize bottom plot
        ax2.set_xlim(0.22, 0.72)
        ax2.set_xticks(np.arange(0.25, 0.75, 0.05))
        ax2.set_xlabel('Background Influence', fontsize=30)
        ax2.tick_params(axis='both', which='major', labelsize=26)
        ax2.set_yticks([])  # Remove y-axis ticks
        ax2.grid(color='gray', linestyle='--', linewidth=2, alpha=0.7)

        # Enhance axis aesthetics
        for spine in ax2.spines.values():
            spine.set_linewidth(2.5)

    # Finalize and save the plot
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/trade_off_{target.replace(' ', '_').lower()}.svg"
    plt.savefig(output_path, transparent=True)
