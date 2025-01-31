import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

# Set the id
id = 'FFY'

# Load data from file
data = np.genfromtxt("23_plots - "+id+".txt", delimiter=',')

# Load questions from file
questions = np.loadtxt("Post questions.txt", dtype=str, delimiter='\t', encoding='utf-8')

# Sort the data according to the first column
sorted_indices = np.argsort(data[:, 0])
data = data[:][sorted_indices]
questions = questions[:][sorted_indices]

def draw_plot(ax, valori, question):
    square_height = 0.35 #Adjust the size of the squares
    fill_height = 4*square_height/5# - 0.01  # Adjust as needed
    position = 2 # Adjust the vertical position of the top square
#def draw_plot(ax, valori, rivalori, question):
#    square_height = 0.6 #Adjust the size of the squares
#    fill_height = 4*square_height/5# - 0.01  # Adjust as needed
#    vposition = 2.77 # Adjust the vertical position of the top square
#    rposition = 1.23

    # Draw rectangle spanning from lower ci to higher ci
    ax.fill_betweenx([position - fill_height / 2, position + fill_height / 2], valori[1], valori[2], color='blue', alpha=0.5,
                     edgecolor='none', linewidth=0)

    # Draw square for top row
    square = Rectangle((valori[0] - square_height / 4, position - square_height / 2), square_height/2, square_height,
                             edgecolor='blue', facecolor='white')
    ax.add_patch(square)

    # Draw rectangle spanning from lower ci to higher ci
 #   ax.fill_betweenx([rposition - fill_height / 2, rposition + fill_height / 2], rivalori[1], rivalori[2], color='green', alpha=0.5,
 #                    edgecolor='none', linewidth=0)

    # Draw square for top row
  #  rsquare = Rectangle((rivalori[0] - square_height / 4, rposition - square_height / 2), square_height/2, square_height,
   #                          edgecolor='green', facecolor='white', label='Second module')
    #ax.add_patch(rsquare)

    # Set y label with multiple lines if necessary
    ax.set_yticks([2])
    ax.set_yticklabels([textwrap.fill(question, width=20)])  # Adjust width as needed

# Create a figure and 23 subplots arranged in a 12x2 grid
fig, axs = plt.subplots(12, 2, figsize=(12, 21), sharex=True, gridspec_kw={'width_ratios': [1, 1]})

# Remove subplot (0, 1)
fig.delaxes(axs[0, 1])

# Draw the first axis
axs[0,0].set_xlim(1, 5)
axs[0,0].set_ylim(0, 4)
#axs[0,0].set_ylim(0.7, 3.3)
axs[0,0].set_xticks(np.arange(1, 6, 4))
axs[0,0].set_xticklabels(['Strongly\ndisagree', 'Strongly\nagree'])

#axs[0,0].xaxis.grid(True, linestyle='--', alpha=0.7)
for val in np.arange(1, 5, 1):
    axs[0,0].axvline(x=val, color='gray', linestyle='--', alpha=0.3)
draw_plot(axs[0,0], data[0,:], questions[0])
#draw_plot(axs[0,0], data[0,:3], data[0,3:], questions[0])

# Draw the rest of the axes
for i in range(1,12):
    axs[i,0].set_xlim(1,5)
    axs[i,0].set_ylim(0,4)
    #axs[i,0].set_ylim(0.7,3.3)
    axs[i,1].set_xlim(1,5)
    axs[i,1].set_ylim(0,4)
    #axs[i,1].set_ylim(0.7,3.3)
	
    # Add vertical light grey dashed lines every 0.2 on the x-axis
    axs[i,0].set_xticks(np.arange(1, 6, 4))
    axs[i,0].set_xticklabels(['Strongly\ndisagree', 'Strongly\nagree'])

    #axs[0,0].xaxis.grid(True, linestyle='--', alpha=0.7)
    for val in np.arange(1, 5, 1):
        axs[i,0].axvline(x=val, color='gray', linestyle='--', alpha=0.3)

    axs[i,1].set_xticks(np.arange(1, 6, 4))
    axs[i,1].set_xticklabels(['Strongly\ndisagree', 'Strongly\nagree'])
    #axs[0,0].xaxis.grid(True, linestyle='--', alpha=0.7)
    for val in np.arange(1, 5, 1):
        axs[i,1].axvline(x=val, color='gray', linestyle='--', alpha=0.3)
	
	# Draw the y-yicks on the right for the second column
    axs[i,1].yaxis.tick_right()
    axs[i,1].yaxis.set_label_position("right")
	
    draw_plot(axs[i,0], data[i,:], questions[i])
    draw_plot(axs[i,1], data[i + int(0.5*len(data[:,0]-1)),:], questions[i + int(0.5*len(data[:,0]-1))])

    #draw_plot(axs[i,0], data[i,:3], data[i,3:], questions[i])
    #draw_plot(axs[i,1], data[i + int(0.5*len(data[:,0]-1)),:3], data[i + int(0.5*len(data[:,0]-1)),3:], questions[i + int(0.5*len(data[:,0]-1))])  

#Set the labels
label_left = axs[11,0].get_xticklabels()
label_left[0].set_ha('center')
label_left[1].set_ha('right')

label_right = axs[11,1].get_xticklabels()
label_right[0].set_ha('left')
label_right[1].set_ha('center')

for ax in axs.flatten():
    ax.tick_params(axis='y', labelsize=11)  # Adjust the labelsize as needed

# Set the title
axs[0,0].set_title('How important for earning a good grade in this class was...', fontsize=19)

# Set x_labels
axs[11,0].set_xlabel('Mean agreement with the statement')
axs[11,1].set_xlabel('Mean agreement with the statement')

#Set legend
#axs[11,1].legend(loc='lower left',prop={'size':16})

# Adjust layout to prevent overlap of axis labels and remove vertical spacing
plt.subplots_adjust(wspace=0.05, hspace=0, left=0.15, right=0.85, top=0.95, bottom=0.05)

# Adjust the position of subplot (0, 0) to be centered above the rest
pos = axs[0, 0].get_position()
pos.x0 = 0.3  # Adjust the horizontal position as needed
pos.x1 = 0.7  # Adjust the horizontal position as needed
axs[0, 0].set_position(pos)

# Save the plot as an image file
plt.savefig('23_plots - '+id+'.png')

# Close the plot to save resources
plt.close()
