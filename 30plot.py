"""
30plot.py - Visualization tool for comparing user and expert responses

This script creates a visualization comparing user responses with expert responses
for a set of 30 questions. It loads data from text files, processes it, and generates
a grid of plots (15 rows, 2 columns) showing the comparison between user ("YOU") and 
expert responses for each question.

Each plot displays:
- Squares representing user (blue) and expert (green) responses
- Arrows showing the direction and magnitude of changes
- Confidence intervals shown as colored regions
- Question text as the y-axis label

The final output is saved as a high-resolution PNG image.

Usage:
    Simply run the script. Make sure the required data files are in the same directory:
    - "30plot - {id}.txt": Contains the response data
    - "Questions.txt": Contains the question text
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

id = 'FFY'  # Identifier used in filenames

# Load response data from file
# The data file contains rows of values representing user and expert responses
data = np.genfromtxt("30plot - " + id + ".txt", delimiter=',')

# Load questions text from file
# Each line contains the text for one question
questions = np.loadtxt("Questions.txt", dtype=str, delimiter='\t', encoding='utf-8')

# Identify even and odd indices in the data array
# Even rows represent user data, odd rows represent expert data
even_indices = np.arange(data.shape[0]) % 2 == 0
odd_indices = ~even_indices

# Sort the data based on the first column of even rows (user data)
# This ensures that questions are displayed in the desired order
sorted_indices = np.argsort(data[even_indices, 0])
data[even_indices] = data[even_indices][sorted_indices]
data[odd_indices] = data[odd_indices][sorted_indices]
questions = questions[:][sorted_indices]


# Function to draw the plot for each row pair
def draw_plot(ax, top_row, bottom_row, question):
    """
    Draw a comparison plot showing user and expert responses for a single question.

    This function creates a visualization that compares user ("YOU") and expert responses
    for a given question. It draws squares representing the responses, arrows showing
    the direction and magnitude of changes, and confidence intervals as colored regions.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object where the plot will be drawn

    top_row : numpy.ndarray
        Array containing data for the user ("YOU") response:
        - top_row[0]: Initial position
        - top_row[1]: Final position
        - top_row[-2]: Lower confidence interval bound
        - top_row[-1]: Upper confidence interval bound

    bottom_row : numpy.ndarray
        Array containing data for the expert response:
        - bottom_row[0]: Initial position
        - bottom_row[1]: Final position
        - bottom_row[-2]: Lower confidence interval bound
        - bottom_row[-1]: Upper confidence interval bound

    question : str
        The text of the question to display as the y-axis label
    """
    square_height = 0.15  # Adjust the size of the squares
    fill_height = 4 * square_height / 5  # - 0.01  # Adjust as needed
    top_position = 0.75  # Adjust the vertical position of the top square
    bottom_position = 0.25  # Adjust the vertical position of the bottom square

    # Draw confidence interval regions as semi-transparent rectangles
    # For the user (top) response: blue rectangle from lower CI to upper CI
    ax.fill_betweenx([top_position - fill_height / 2, top_position + fill_height / 2], top_row[-2], top_row[-1],
                     color='blue', alpha=0.5,
                     edgecolor='none', linewidth=0)
    # For the expert (bottom) response: green rectangle from lower CI to upper CI
    ax.fill_betweenx([bottom_position - fill_height / 2, bottom_position + fill_height / 2], bottom_row[-2],
                     bottom_row[-1], color='green', alpha=0.5,
                     edgecolor='none', linewidth=0)

    # Draw square representing the user's initial position (blue outline)
    # This square is positioned at top_row[0] on the x-axis and at top_position on the y-axis
    square_top = Rectangle((top_row[0] - square_height / 4, top_position - square_height / 2), square_height / 2,
                           square_height,
                           edgecolor='blue', facecolor='white', label='YOU')
    ax.add_patch(square_top)

    # Draw square representing the expert's initial position (green outline)
    # This square is positioned at bottom_row[0] on the x-axis and at bottom_position on the y-axis
    square_bottom = Rectangle((bottom_row[0] - square_height / 4, bottom_position - square_height / 2),
                              square_height / 2, square_height,
                              edgecolor='green', facecolor='white', label='EXPERT')
    ax.add_patch(square_bottom)

    # Define the size of arrowheads for the arrows
    top_arrowhead_size = 0.02
    bottom_arrowhead_size = 0.02

    # Calculate arrow length for user (top) response
    # The arrow shows the change from initial position (top_row[0]) to final position (top_row[1])
    # We need to adjust the arrow length to account for the arrowhead
    if abs(top_row[1] - top_row[0]) >= top_arrowhead_size:
        if top_row[1] - top_row[0] >= 0:  # Moving right
            top_arrow_length = top_row[1] - top_row[0] - top_arrowhead_size
        else:  # Moving left
            top_arrow_length = top_row[1] - top_row[0] + top_arrowhead_size
    else:  # Very small change, use minimal arrow length
        top_arrow_length = 0.000001
        top_arrowhead_size = top_row[1] - top_row[0]

    # Calculate arrow length for expert (bottom) response
    # Similar logic as for the user response
    if abs(bottom_row[1] - bottom_row[0]) >= bottom_arrowhead_size:
        if bottom_row[1] - bottom_row[0] >= 0:  # Moving right
            bottom_arrow_length = bottom_row[1] - bottom_row[0] - bottom_arrowhead_size
        else:  # Moving left
            bottom_arrow_length = bottom_row[1] - bottom_row[0] + bottom_arrowhead_size
    else:  # Very small change, use minimal arrow length
        bottom_arrow_length = 0.0000001
        bottom_arrowhead_size = bottom_row[1] - bottom_row[0]

    # Draw the arrow for user (top) response
    ax.arrow(top_row[0], top_position, top_arrow_length, 0, head_width=top_arrowhead_size,
             head_length=top_arrowhead_size, fc='black', ec='black', zorder=10)

    # Draw the arrow for expert (bottom) response
    ax.arrow(bottom_row[0], bottom_position, bottom_arrow_length, 0, head_width=bottom_arrowhead_size,
             head_length=bottom_arrowhead_size, fc='black', ec='black', zorder=10)

    # Set the question text as the y-axis label
    # Position the label in the middle of the y-axis (at 0.5)
    # Use textwrap to wrap long question text to multiple lines for better readability
    ax.set_yticks([0.5])
    ax.set_yticklabels([textwrap.fill(question, width=22)])  # Width of 22 characters per line


# Create a single figure with 30 subplots (15 rows, 2 columns)
# This will display all 30 questions in a grid layout
fig, axes = plt.subplots(15, 2, figsize=(10, 20), sharex=True, gridspec_kw={'width_ratios': [1, 1]})

# Iterate over the data and draw subplots
# We process 2 rows at a time (user and expert data for each question)
for i in range(0, int(len(data[:, 0]) / 2), 2):
    # Get the axes for the current row (left and right columns)
    ax1 = axes[i // 2, 0]  # Left column
    ax2 = axes[i // 2, 1]  # Right column

    # Draw plot for the first 15 questions in the left column
    draw_plot(ax1, data[i, :], data[i + 1, :], questions[i // 2])
    ax1.set_xlim(0, 1)  # Set x-axis range from 0 to 1 (fraction of responses)
    ax1.set_ylim(0, 1)  # Set y-axis range

    # Draw plot for the second 15 questions in the right column
    draw_plot(ax2, data[i + 30, :], data[i + 31, :], questions[i // 2 + 15])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Add vertical light grey dashed grid lines every 0.2 on the x-axis
    # This helps with reading the values from the plot
    ax1.set_xticks(np.arange(0, 1.2, 0.2))
    ax1.set_xticklabels([f'{i:.1f}' for i in np.arange(0, 1.2, 0.2)])
    ax1.xaxis.grid(True, linestyle='--', alpha=0.7)

    ax2.set_xticks(np.arange(0, 1.2, 0.2))
    ax2.set_xticklabels([f'{i:.1f}' for i in np.arange(0, 1.2, 0.2)])
    ax2.xaxis.grid(True, linestyle='--', alpha=0.7)

    # Position the y-axis ticks on the right side for the right column
    # This improves readability when questions are displayed on both sides
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

# Add a legend to the last plot in the grid
# This explains the color coding (blue for user, green for expert)
axes[14, 1].legend()

# Add x-axis labels only to the bottom plots
# This avoids cluttering the figure with redundant labels
axes[14, 0].set_xlabel('Fraction of class with expert-like response')
axes[14, 1].set_xlabel('Fraction of class with expert-like response')

# Set the main title for the entire figure
axes[0, 0].set_title('What do YOU think? vs. what do EXPERTS think?', y=1.0, x=1.05, fontsize=21)

# Adjust the margins to ensure all elements fit properly
plt.subplots_adjust(bottom=0.15, top=0.8)

# Set consistent font size for all y-tick labels (question text)
for ax in axes.flatten():
    ax.tick_params(axis='y', labelsize=10)

# Optimize the layout and adjust spacing between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.0, wspace=0.1)  # No vertical space, minimal horizontal space

# Save the figure as a high-resolution PNG image
fig.savefig('30plot - ' + id + '.png', format='png', dpi=400)
plt.close()  # Close the figure to free up memory resources
