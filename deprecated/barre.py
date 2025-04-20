import numpy as np
import matplotlib.pyplot as plt

# Fix the id
id = 'FFY'

# Load data from the txt file
data = np.loadtxt("barre_"+id+".txt", delimiter="\t")

# Fix our entries
data[:] = data[:]/30

# Extract data for "YOU" and "Expert" categories
pre_data = data[0, :2]  # First three columns for "PRE"
pre_error = data[1, :2]  # Error values for "PRE"
#mean_data = data[0, 2:4]
#mean_error = data[1, 2:4]
#post_data = data[0, 4:]  # Next three columns for "POST"
#post_error = data[1, 4:]  # Error values for "POST"
post_data = data[0, 2:]  # Next three columns for "POST"
post_error = data[1, 2:]  # Error values for "POST"

# Define x values and labels
x_values = np.array([0,1]) # Two groups: "YOU" and "Expert"
labels = ["Pre", "Post"]
#x_values = np.array([0,1,2]) # Two groups: "YOU" and "Expert"
#labels = ["Pre", "Intermediate", "Post"]

# Define colors for bars
colors = ['blue', 'green']

# Function to add values on top of bars
def add_values_on_top(bars, fontsize=8.5):
    for i in range(len(bars)):
        bar = bars[i][0][0]
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + bars[i][1],
            f"{height:.2f}",
            ha="center",
            va="bottom",
			fontsize=fontsize
        )

# Creating a list for the bars
bar_list = []

# Plotting
fig, ax = plt.subplots()

# Plot bars for "Pre" data
bar_list.append([ax.bar(x_values[0] - 0.1, pre_data[0], yerr=pre_error[0], width=0.2, color=colors[0], edgecolor='black', alpha=0.7, label="YOU"), pre_error[0]])
bar_list.append([ax.bar(x_values[0] + 0.1, pre_data[1], yerr=pre_error[1], width=0.2, color=colors[1], edgecolor='black', alpha=0.7, label="EXPERT"), pre_error[1]])

# Plot bars for "Post" data
#bar_list.append([ax.bar(x_values[1] - 0.1, mean_data[0], yerr=mean_error[0], width=0.2, color=colors[0], edgecolor='black', alpha=0.7), mean_error[0]])
#bar_list.append([ax.bar(x_values[1] + 0.1, mean_data[1], yerr=mean_error[1], width=0.2, color=colors[1], edgecolor='black', alpha=0.7), mean_error[1]])

# Plot bars for "Post" data
#bar_list.append([ax.bar(x_values[2] - 0.1, post_data[0], yerr=post_error[0], width=0.2, color=colors[0], edgecolor='black', alpha=0.7), post_error[0]])
#bar_list.append([ax.bar(x_values[2] + 0.1, post_data[1], yerr=post_error[1], width=0.2, color=colors[1], edgecolor='black', alpha=0.7), post_error[1]])

# Plot bars for "Post" data
bar_list.append([ax.bar(x_values[1] - 0.1, post_data[0], yerr=post_error[0], width=0.2, color=colors[0], edgecolor='black', alpha=0.7), post_error[0]])
bar_list.append([ax.bar(x_values[1] + 0.1, post_data[1], yerr=post_error[1], width=0.2, color=colors[1], edgecolor='black', alpha=0.7), post_error[1]])

# Add error labels on top of bars
add_values_on_top(bar_list)

# Set labels and title
ax.set_xticks(x_values)
ax.set_xticklabels(labels)
ax.set_ylabel("Fraction of statements\nwith expert-like response")
ax.set_title('Overall E-CLASS score on "What do YOU think..."\nand "What do EXPERTS think..." statements')
ax.set_ylim(0, 1)

# Add legend
ax.legend(loc='lower center')

# Save the plot
fig.savefig('Barre - '+id+'.png', format='png', dpi=400)
plt.close()  # Close the figure to free up resources