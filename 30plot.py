import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

id = 'FFY'

# Load data from file
data = np.genfromtxt("30plot - "+id+".txt", delimiter=',')

# Load questions from file
questions = np.loadtxt("Questions.txt", dtype=str, delimiter='\t', encoding='utf-8')

# Identify even and odd indices
even_indices = np.arange(data.shape[0]) % 2 == 0
odd_indices = ~even_indices

# Sort even rows based on the first column
sorted_indices = np.argsort(data[even_indices, 0])
data[even_indices] = data[even_indices][sorted_indices]
data[odd_indices] = data[odd_indices][sorted_indices]
questions = questions[:][sorted_indices]

# Function to draw the plot for each row pair
def draw_plot(ax, top_row, bottom_row, question):
    square_height = 0.15 #Adjust the size of the squares
    fill_height = 4*square_height/5# - 0.01  # Adjust as needed
    top_position = 0.75 # Adjust the vertical position of the top square
    bottom_position = 0.25 # Adjust the vertical position of the bottom square

    # Draw rectangle spanning from lower ci to higher ci
    ax.fill_betweenx([top_position - fill_height / 2, top_position + fill_height / 2], top_row[-2], top_row[-1], color='blue', alpha=0.5,
                     edgecolor='none', linewidth=0)
    ax.fill_betweenx([bottom_position - fill_height / 2, bottom_position + fill_height / 2], bottom_row[-2], bottom_row[-1], color='green', alpha=0.5,
                     edgecolor='none', linewidth=0)

    # Draw square for top row
    square_top = Rectangle((top_row[0] - square_height / 4, top_position - square_height / 2), square_height/2, square_height,
                             edgecolor='blue', facecolor='white', label='YOU')
    ax.add_patch(square_top)

    # Draw square for bottom row
    square_bottom = Rectangle((bottom_row[0] - square_height / 4, bottom_position - square_height / 2), square_height/2, square_height,
                               edgecolor='green', facecolor='white', label='EXPERT')
    ax.add_patch(square_bottom)
	
    # Draw horizontal arrow from square to post mean with arrowhead
 #-   arrowhead_size = 0.02


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


    ax.arrow(top_row[0], top_position, top_arrow_length, 0, head_width=top_arrowhead_size, head_length=top_arrowhead_size, fc='black', ec='black', zorder=10)
    ax.arrow(bottom_row[0], bottom_position, bottom_arrow_length, 0, head_width=bottom_arrowhead_size, head_length=bottom_arrowhead_size, fc='black', ec='black', zorder=10)

#    ax.arrow(top_row[0], top_position, top_arrow_length, 0, head_width=top_arrowhead_size, head_length=top_arrowhead_size, fc='red', ec='red', zorder=10)
#    ax.arrow(bottom_row[0], bottom_position, bottom_arrow_length, 0, head_width=bottom_arrowhead_size, head_length=bottom_arrowhead_size, fc='red', ec='red', zorder=10)

#    top_arrowhead_size = 0.02
#    bottom_arrowhead_size = 0.02
	
	
 #   if abs(top_row[2] - top_row[1]) >= top_arrowhead_size:
  #      if top_row[2] - top_row[1] >= 0:
   #         top_arrow_length = top_row[2] - top_row[1] - top_arrowhead_size
    #    else:
    #        top_arrow_length = top_row[2] - top_row[1] + top_arrowhead_size
    #else:
 #       top_arrow_length = 0.000001
 #       top_arrowhead_size = top_row[2] - top_row[1]

  #  if abs(bottom_row[2] - bottom_row[1]) >= bottom_arrowhead_size:
   #     if bottom_row[2] - bottom_row[1] >= 0:
  #          bottom_arrow_length = bottom_row[2] - bottom_row[1] - bottom_arrowhead_size
  #      else:
  #          bottom_arrow_length = bottom_row[2] - bottom_row[1] + bottom_arrowhead_size
  #  else:
  #      bottom_arrow_length = 0.0000001
  #      bottom_arrowhead_size = bottom_row[2] - bottom_row[1]

 #   ax.arrow(top_row[1], top_position, top_arrow_length, 0, head_width=top_arrowhead_size, head_length=top_arrowhead_size, fc='black', ec='black', zorder=10)
 #   ax.arrow(bottom_row[1], bottom_position, bottom_arrow_length, 0, head_width=bottom_arrowhead_size, head_length=bottom_arrowhead_size, fc='black', ec='black', zorder=10)

	
    # Set y label with multiple lines if necessary
    ax.set_yticks([0.5])
    ax.set_yticklabels([textwrap.fill(question, width=22)])  # Adjust width as needed

# Create a single figure with 30 subplots
fig, axes = plt.subplots(15, 2, figsize=(10, 20), sharex=True, gridspec_kw={'width_ratios': [1, 1]})

# Iterate over the data and draw subplots
for i in range(0, int(len(data[:,0])/2), 2):
    ax1 = axes[i//2, 0]
    ax2 = axes[i//2, 1]

    draw_plot(ax1, data[i,:], data[i + 1,:], questions[i//2])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    draw_plot(ax2, data[i + 30,:], data[i + 31,:], questions[i//2 + 15])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Add vertical light grey dashed lines every 0.2 on the x-axis
    ax1.set_xticks(np.arange(0, 1.2, 0.2))
    ax1.set_xticklabels([f'{i:.1f}' for i in np.arange(0, 1.2, 0.2)])
    ax1.xaxis.grid(True, linestyle='--', alpha=0.7)

    ax2.set_xticks(np.arange(0, 1.2, 0.2))
    ax2.set_xticklabels([f'{i:.1f}' for i in np.arange(0, 1.2, 0.2)])
    ax2.xaxis.grid(True, linestyle='--', alpha=0.7)
	
	# Draw the y-yicks on the right for the second column
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

# Draw legend only for the last graph
axes[14, 1].legend()

# Draw x-label for the bottom graph only
axes[14, 0].set_xlabel('Fraction of class with expert-like response')
axes[14, 1].set_xlabel('Fraction of class with expert-like response')

# Set the title
axes[0,0].set_title('What do YOU think? vs. what do EXPERTS think?', y=1.0, x=1.05, fontsize=21)

plt.subplots_adjust(bottom=0.15, top=0.8)  # Adjust the bottom margin as needed

# Set y-tick font size for all subplots
for ax in axes.flatten():
    ax.tick_params(axis='y', labelsize=10)  # Adjust the labelsize as needed

# Adjust layout and vertical spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.0, wspace=0.1)  # Adjust horizontal spacing as needed

# Save figure to the "results" folder
fig.savefig('30plot - '+id+'.png', format='png', dpi=400)
plt.close()  # Close the figure to free up resources