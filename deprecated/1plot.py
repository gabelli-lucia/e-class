import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

id = 'FSY'
i = 29

# Load data from file
data = np.genfromtxt("30plot - " + id + ".txt", delimiter=',')

# Load questions from file
questions = np.loadtxt("Questions.txt", dtype=str, delimiter='\t', encoding='utf-8')

# Identify even and odd indices
# even_indices = np.arange(data.shape[0]) % 2 == 0
# odd_indices = ~even_indices

# Sort even rows based on the first column
# sorted_indices = np.argsort(data[even_indices, 0])
# data[even_indices] = data[even_indices][sorted_indices]
# data[odd_indices] = data[odd_indices][sorted_indices]
# questions = questions[:][sorted_indices]

# Function to draw the plot for each row pair
def draw_plot(ax, top_row, bottom_row, question):
    square_height = 0.083  # Adjust the size of the squares
    fill_height = 4 * square_height / 5  # - 0.01  # Adjust as needed
    top_position = 0.75  # Adjust the vertical position of the top square
    bottom_position = 0.25  # Adjust the vertical position of the bottom square

    # Draw rectangle spanning from lower ci to higher ci
    ax.fill_betweenx([top_position - fill_height / 2, top_position + fill_height / 2], top_row[-2], top_row[-1],
                     color='blue', alpha=0.5,
                     edgecolor='none', linewidth=0)
    ax.fill_betweenx([bottom_position - fill_height / 2, bottom_position + fill_height / 2], bottom_row[-2],
                     bottom_row[-1], color='green', alpha=0.5,
                     edgecolor='none', linewidth=0)

    # Draw square for top row
    square_top = Rectangle((top_row[0] - square_height / 4, top_position - square_height / 2), square_height / 2,
                           square_height,
                           edgecolor='blue', facecolor='white', label='YOU')
    ax.add_patch(square_top)

    # Draw square for bottom row
    square_bottom = Rectangle((bottom_row[0] - square_height / 4, bottom_position - square_height / 2),
                              square_height / 2, square_height,
                              edgecolor='green', facecolor='white', label='EXPERT')
    ax.add_patch(square_bottom)

    # Draw horizontal arrow from square to post mean with arrowhead
    # -   arrowhead_size = 0.02

    # if abs(top_row[1]-top_row[0]) > square_height/4:
    #    if top_row[1]-top_row[0] > 0:
    #       ax.arrow(top_row[0], top_position, top_row[1]-top_row[0] - arrowhead_size, 0, head_width=arrowhead_size, head_length=arrowhead_size, fc='black', ec='black', zorder=10)
    #  else:
    #     ax.arrow(top_row[0], top_position, top_row[1]-top_row[0] + arrowhead_size, 0, head_width=arrowhead_size, head_length=arrowhead_size, fc='black', ec='black', zorder=10)

    #    if abs(bottom_row[1]-bottom_row[0]) > square_height/4:
    #       if bottom_row[1]-bottom_row[0] > 0:
    #          ax.arrow(bottom_row[0], bottom_position, bottom_row[1]-bottom_row[0] - arrowhead_size, 0, head_width=arrowhead_size, head_length=arrowhead_size, fc='black', ec='black', zorder=10)
    #     else:
    #        ax.arrow(bottom_row[0], bottom_position, bottom_row[1]-bottom_row[0] + arrowhead_size, 0, head_width=arrowhead_size, head_length=arrowhead_size, fc='black', ec='black', zorder=10)

    top_arrowhead_size = 0.02
    bottom_arrowhead_size = 0.02

    if abs(top_row[1] - top_row[0]) >= top_arrowhead_size:
        if top_row[1] - top_row[0] >= 0:
            top_arrow_length = top_row[1] - top_row[0] - top_arrowhead_size
        else:
            top_arrow_length = top_row[1] - top_row[0] + top_arrowhead_size
    else:
        top_arrow_length = 0.000001
        top_arrowhead_size = top_row[1] - top_row[0]

    if abs(bottom_row[1] - bottom_row[0]) >= bottom_arrowhead_size:
        if bottom_row[1] - bottom_row[0] >= 0:
            bottom_arrow_length = bottom_row[1] - bottom_row[0] - bottom_arrowhead_size
        else:
            bottom_arrow_length = bottom_row[1] - bottom_row[0] + bottom_arrowhead_size
    else:
        bottom_arrow_length = 0.0000001
        bottom_arrowhead_size = bottom_row[1] - bottom_row[0]

    # ax.arrow(top_row[0], top_position, top_arrow_length, 0, head_width=top_arrowhead_size, head_length=top_arrowhead_size, fc='black', ec='black', zorder=10)
    # ax.arrow(bottom_row[0], bottom_position, bottom_arrow_length, 0, head_width=bottom_arrowhead_size, head_length=bottom_arrowhead_size, fc='black', ec='black', zorder=10)

    ax.arrow(top_row[0], top_position, top_arrow_length, 0, head_width=top_arrowhead_size,
             head_length=top_arrowhead_size, fc='red', ec='red', zorder=10)
    ax.arrow(bottom_row[0], bottom_position, bottom_arrow_length, 0, head_width=bottom_arrowhead_size,
             head_length=bottom_arrowhead_size, fc='red', ec='red', zorder=10)

    top_arrowhead_size = 0.02
    bottom_arrowhead_size = 0.02

    if abs(top_row[2] - top_row[1]) >= top_arrowhead_size:
        if top_row[2] - top_row[1] >= 0:
            top_arrow_length = top_row[2] - top_row[1] - top_arrowhead_size
        else:
            top_arrow_length = top_row[2] - top_row[1] + top_arrowhead_size
    else:
        top_arrow_length = 0.000001
        top_arrowhead_size = top_row[2] - top_row[1]

    if abs(bottom_row[2] - bottom_row[1]) >= bottom_arrowhead_size:
        if bottom_row[2] - bottom_row[1] >= 0:
            bottom_arrow_length = bottom_row[2] - bottom_row[1] - bottom_arrowhead_size
        else:
            bottom_arrow_length = bottom_row[2] - bottom_row[1] + bottom_arrowhead_size
    else:
        bottom_arrow_length = 0.0000001
        bottom_arrowhead_size = bottom_row[2] - bottom_row[1]

    ax.arrow(top_row[1], top_position, top_arrow_length, 0, head_width=top_arrowhead_size,
             head_length=top_arrowhead_size, fc='black', ec='black', zorder=10)
    ax.arrow(bottom_row[1], bottom_position, bottom_arrow_length, 0, head_width=bottom_arrowhead_size,
             head_length=bottom_arrowhead_size, fc='black', ec='black', zorder=10)

    # Set y label with multiple lines if necessary
    ax.set_yticks([])
    # ax.set_yticklabels([textwrap.fill(question, width=22)])  # Adjust width as needed


fig, ax = plt.subplots()
draw_plot(ax, data[2 * (i - 1), :], data[2 * i - 1, :], questions[i - 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
# Add vertical light grey dashed lines every 0.2 on the x-axis
ax.set_xticks(np.arange(0, 1.2, 0.2))
ax.set_xticklabels([f'{i:.1f}' for i in np.arange(0, 1.2, 0.2)])
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel('Fraction of class with expert-like response')
# ax.set_title(textwrap.fill(questions[i-1], width=65))
# plt.subplots_adjust(top=0.85) #per titoli lunghi
# ax.tick_params(axis='y', labelsize=10)  # Adjust the labelsize as needed
# Save figure to the "results" folder
fig.savefig('1plot - ' + id + ' ' + str(i) + '.png', format='png', dpi=400)
plt.close()  # Close the figure to free up resources
