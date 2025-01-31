import numpy as np
import matplotlib.pyplot as plt

id = "FFY"

# Load data from "significance.txt"
data = np.genfromtxt("significance_FFY.txt", delimiter=",", dtype=str)
#data1 = np.genfromtxt("significance_secondo_anno_pre_post.txt", delimiter=",", dtype=str)
#data2 = np.genfromtxt("significance_secondo_anno_post_post.txt", delimiter=",", dtype=str)

# Extract relevant columns
selected_columns = list(range(0, 60, 2)) + [-4, -3, -2, -1]
filtered_data = data[:, selected_columns]
filtered_data = filtered_data[filtered_data[:,-1] == 'Matched', :]
#filtered_data1 = data1[:, selected_columns]
#filtered_data1 = filtered_data1[filtered_data1[:,-1] == 'Matched', :]
#filtered_data2 = data2[:, selected_columns]
#filtered_data2 = filtered_data2[filtered_data2[:,-1] == 'Matched', :]

# Separate data into PRE and POST categories
pre_data = filtered_data[filtered_data[:, -2] == "PRE"]
post_data = filtered_data[filtered_data[:, -2] == "POST"]
#pre_data = filtered_data1[filtered_data1[:, -2] == "PRE"]
#mean_data = filtered_data2[filtered_data2[:, -2] == "PRE"]
#post_data = filtered_data2[filtered_data2[:, -2] == "POST"]

# Separate data into Physics and Engineering
#pre = pre_data[pre_data[:, -3] == id]
#post = post_data[post_data[:, -3] == id]

def grafico(data_i, data_f, cohen, nome='Significance.png'):
#def grafico(data_i, data_m, data_f, cohen, nome='Significance.png'):

    # Calculate average of each column
    pre_avg = np.mean(data_i[:, :-4].astype(float), axis=0)
    post_avg = np.mean(data_f[:, :-4].astype(float), axis=0)
#    mean_avg = np.mean(data_m[:, :-4].astype(float), axis=0)

    # Combine x and y values for sorting
    pre_sorted = np.column_stack((np.arange(len(pre_avg)), pre_avg))
    post_sorted = np.column_stack((np.arange(len(post_avg)), post_avg))
#    mean_sorted = np.column_stack((np.arange(len(mean_avg)), mean_avg))

    # Sort data by y values
    sorted_indices = pre_sorted[:, 1].argsort()
    pre_sorted = pre_sorted[sorted_indices]
    post_sorted = post_sorted[sorted_indices]
 #   mean_sorted = mean_sorted[sorted_indices]

    # Set figure size
    plt.figure(figsize=(10, 6))  # Adjust width and height as needed

    # Plot the data
    plt.plot(np.arange(len(pre_avg)), pre_sorted[:, 1], marker='s', markersize=8, linestyle='--', linewidth=0.8, label='PRE', color='blue', markerfacecolor='white')
 #   plt.plot(np.arange(len(mean_avg)), mean_sorted[:, 1], marker='^', markersize=8, linestyle='--', linewidth=0.8, label='INT', color='red')
    plt.plot(np.arange(len(pre_avg)), post_sorted[:, 1], marker='o', markersize=8, linestyle='--', linewidth=0.8, label='POST', color='green')

    # Get the primary axis
    ax1 = plt.gca()

    # Set x-ticks to the original index values
    plt.xticks(np.arange(len(pre_avg)), sorted_indices + 1)

    # Add vertical lines
    for tick in sorted_indices:
        plt.axvline(x=tick, color='lightgray', linestyle='-', linewidth=0.5, zorder=1)

    # Add labels and legend
    plt.xlabel('Question', fontsize=16)
    plt.legend()

    # Import data from "d_values.txt" for vertical bars
    d_values = np.genfromtxt(cohen, delimiter=",")
    if np.size(d_values) > 0 and len(d_values.shape) == 1:
        # If it's a single row, convert it to a 2D array with a single row
        d_values = np.array([d_values])

    # Create a secondary y-axis for Cohen's d values on the right with custom limits
    ax2 = plt.gca().twinx()
    ax2.set_ylim([0, 1])  # Adjust the limits as needed

    if np.size(d_values) > 0:
        for d in d_values:
            x_value = np.arange(len(pre_avg))[sorted_indices == d[0] - 1]  # Find the match in the sorted indices
            y_value = abs(d[1])
            y_value = abs(d[1])
            ax2.bar(x_value, y_value, color='gray', alpha=0.35, zorder=1)

    	    # Add a black star slightly above the highest marker
            ax1.plot(x_value, max(pre_sorted[x_value, 1], post_sorted[x_value, 1]) + 0.1, marker='*', markersize=10, color='black')

    # Set the label for the primary y-axis
    ax1.set_ylabel('Average Agreement with Experts', fontsize=16)

    # Set the label for the secondary y-axis
    ax2.set_ylabel('Effect Size', fontsize=16)

    plt.title('Average score on "What do YOU think?" questions', fontsize=19)

    # Save the plot
    plt.savefig('Significance - '+id+'.png', dpi=400)
	
    # Show the plot
    plt.close()

grafico(pre_data, post_data, 'd_values - '+id+'.txt', nome=id)
#grafico(pre_data, mean_data, post_data, 'd_values - '+id+'.txt', nome=id)