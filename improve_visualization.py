import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the evaluation summary CSV
df = pd.read_csv('evaluation_summary.csv')

# Sort by accuracy for consistent ordering
df = df.sort_values(by='Accuracy', ascending=False)

# Set up solid colors palette (no gradient)
solid_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf'   # cyan
]

# Create a color dictionary for consistent colors across plots
color_dict = {model: color for model, color in zip(df['Model'], solid_colors)}

# Common settings for all plots
def plot_bar(ax, data, title, ylabel, color_dict):
    bars = ax.bar(range(len(data)), data, color=[color_dict[model] for model in df['Model']])
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(data):
        ax.text(i, v + max(data)*0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

# ==================== FIGURE 1: ACCURACY METRICS ====================
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
fig1.suptitle('Model Accuracy Metrics', fontsize=16, y=0.98)

# 1. Accuracy (Semantic Similarity)
plot_bar(axes1[0], df['Accuracy'].values, 'Accuracy (Semantic Similarity)', 'Score', color_dict)

# 2. Faithfulness
plot_bar(axes1[1], df['Faithfulness'].values, 'Faithfulness (LLM as Judge)', 'Score', color_dict)

# Adjust layout
fig1.tight_layout(rect=[0, 0, 1, 0.95])

# ==================== FIGURE 2: SPEED METRICS ====================
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Model Speed Metrics', fontsize=16, y=0.98)

# 3. Response Time
plot_bar(axes2[0], df['Avg. Time (s)'].values, 'Average Response Time', 'Seconds', color_dict)

# 4. Tokens per Second
plot_bar(axes2[1], df['Tokens/s'].values, 'Tokens per Second', 'Tokens/s', color_dict)

# Adjust layout
fig2.tight_layout(rect=[0, 0, 1, 0.95])

# ==================== FIGURE 3: ANSWER TYPES (HORIZONTAL STACKED BAR) ====================
fig3, ax3 = plt.subplots(figsize=(12, 8))
fig3.suptitle('Answer Types by Model', fontsize=16, y=0.98)

# Prepare data for horizontal stacked bar chart
models = df['Model'].values
correct_counts = df['Correct Answers'].values
incorrect_counts = df['Incorrect Answers'].values
partial_counts = df['Partial Answers'].values
no_info_counts = df['No Information'].values

# Create the horizontal stacked bars
y_pos = np.arange(len(models))
height = 0.8

# Plot each segment
ax3.barh(y_pos, correct_counts, height, label='Correct Answers', color='#2ca02c')

# Calculate the left positions for each segment
left_for_incorrect = correct_counts.copy()
ax3.barh(y_pos, incorrect_counts, height, left=left_for_incorrect, label='Incorrect Answers', color='#d62728')

left_for_partial = [correct + incorrect for correct, incorrect in zip(correct_counts, incorrect_counts)]
ax3.barh(y_pos, partial_counts, height, left=left_for_partial, label='Partial Answers', color='#ff7f0e')

left_for_no_info = [correct + incorrect + partial for correct, incorrect, partial in zip(correct_counts, incorrect_counts, partial_counts)]
ax3.barh(y_pos, no_info_counts, height, left=left_for_no_info, label='No Information', color='#7f7f7f')

# Set title and labels
ax3.set_title('Answer Types by Model', fontsize=14, pad=10)
ax3.set_xlabel('Number of Responses', fontsize=12)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(models, fontsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='x', linestyle='--', alpha=0.7)

# Add a legend
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

# Add value labels inside each segment if there's enough space
for i in range(len(models)):

    # Add labels inside each segment if there's enough space
    if correct_counts[i] > 1:
        ax3.text(correct_counts[i]/2, i, f'{correct_counts[i]}', 
                va='center', ha='center', fontsize=9, color='white')
    
    if incorrect_counts[i] > 1:
        ax3.text(left_for_incorrect[i] + incorrect_counts[i]/2, i, f'{incorrect_counts[i]}', 
                va='center', ha='center', fontsize=9, color='white')
    
    if partial_counts[i] > 1:
        ax3.text(left_for_partial[i] + partial_counts[i]/2, i, f'{partial_counts[i]}', 
                va='center', ha='center', fontsize=9, color='white')
    
    if no_info_counts[i] > 1:
        ax3.text(left_for_no_info[i] + no_info_counts[i]/2, i, f'{no_info_counts[i]}', 
                va='center', ha='center', fontsize=9, color='white')

# Adjust layout
fig3.tight_layout(rect=[0, 0, 1, 0.95])

# Save all figures
fig1.savefig('accuracy_metrics.png', dpi=300, bbox_inches='tight')
fig2.savefig('speed_metrics.png', dpi=300, bbox_inches='tight')
fig3.savefig('answer_types.png', dpi=300, bbox_inches='tight')