import csv
import numpy as np
import matplotlib.pyplot as plt
import textwrap

id = 'B-sep'

# Load data from CSV file
data = np.genfromtxt('cluster'+id+'.txt', delimiter=',')
data=data.T
#data = np.genfromtxt(id+'.txt')

# Split data into separate histograms
histograms = [
    data[:3],    # Next three columns
    data[3:9],   # Next six columns
    data[9:16],  # Next seven columns
    data[16:20], # Next four columns
    data[20:24],    # Next four columns
    data[24:]    # Last three columns
]

# Retrieve the column labels
words = np.loadtxt("titles.txt", dtype=str, delimiter='\t', encoding='utf-8')
column_labels = []
column_labels.append(list(words[:3]))
column_labels.append(list(words[3:9]))
column_labels.append(list(words[9:16]))
column_labels.append(list(words[16:20]))
column_labels.append(list(words[20:24]))
column_labels.append(list(words[24:]))

# Set the titles of the plot
titles = ['Type of investigation: Students...', 'Student agency: Students...','Development and use of models: Students...','Data analysis and visualisation: Students...', \
   'Communication: Students...','How important are the following'+'\n'+'macro-goals in structuring the lab activities?']

# Set the y-ticks for the first figure
first_y_ticks = ["Never","Rarely","Sometimes","Often","Always"]
last_y_ticks = ["It's not\na goal", "Less\nthan 50%", "50%", "More\nthan 50%", "It's the\nonly goal"]

# Function to create and save bar plot
#def create_and_save_bar_plot(data, errore, x_labels, title, filename, mult, size, wrap, bott, index):
#    plt.bar([mult*value for value in range(len(data))], data, yerr=errore, capsize=5, error_kw={'elinewidth': 0.75, 'capthick': 0.8}, width=mult*0.8)
#def create_and_save_bar_plot(data, x_labels, title, filename, mult, size, wrap, bott, index):
#    plt.bar([mult*value for value in range(len(data))], data, width=mult*0.8)
def create_and_save_bar_plot(data, x_labels, title, filename, mult, size, wrap, bott, index):
#    a = [mult*value -(.8/3+0.009)*mult for value in range(len(data[:,0]))]
#    b = [mult*value +(.8/3+0.009)*mult for value in range(len(data[:,2]))]
#    plt.bar(a, data[:,0], width=mult*0.8/3)
#    plt.bar([mult*value for value in range(len(data[:,1]))], data[:,1], width=mult*0.8/3)
#    plt.bar(b, data[:,2], width=mult*0.8/3)
    a = [mult*value -.205*mult for value in range(len(data[:,0]))]
    b = [mult*value +.205*mult for value in range(len(data[:,1]))]
    plt.bar(a, data[:,0], width=mult*0.4)
    plt.bar(b, data[:,1], width=mult*0.4)
    plt.title(title)
    if wrap == 19:
        ticks = []
        for i in range(len(x_labels)):
            if i!=4:
                ticks.append('\n'.join(textwrap.wrap(x_labels[i], width=wrap-3)))
            else:
                ticks.append('\n'.join(textwrap.wrap(x_labels[i], width=wrap)))
        plt.xticks([mult*value for value in range(len(data))], ticks, fontsize = size)  # Set x-tick labels with rotation
    else:
        plt.xticks([mult*value for value in range(len(data))], ['\n'.join(textwrap.wrap(label, width=wrap)) for label in x_labels], fontsize = size)  # Set x-tick labels with rotation
    if index !=5:
        plt.yticks([1,2,3,4,5], first_y_ticks, fontsize=8)  # Set y-tick labels
    else:
        #plt.yticks(list(string_to_value.values()), ['\n'.join(textwrap.wrap(label, width=9)) for label in first_y_ticks], fontsize=8)
        plt.yticks([1,2,3,4,5], last_y_ticks, fontsize=8)
    plt.subplots_adjust(bottom=bott)  # Adjust the bottom margin as needed
    plt.gca().set_ylim(1,5.2)
    plt.savefig(filename, dpi=400)
    plt.close()

# Create and save bar plot for each element of histograms
for i, histogram_data in enumerate(histograms):
    # Select corresponding column labels for x-axis
    filename = f"Cluster {id} - Prof {i+1}.png"
 #   filename = f"{id} - Prof {i+1}.png"
    if i==0:
 #       create_and_save_bar_plot(histogram_data[:,0], histogram_data[:,1], column_labels[i], titles[i], filename, 1, 8, 20, 0.175, i)
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 1, 8, 20, 0.175, i)
    elif i==1:
#        create_and_save_bar_plot(histogram_data[:,0], histogram_data[:,1], column_labels[i], titles[i], filename, 1, 7.5, 12, 0.175, i)
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 1, 7.5, 12, 0.175, i)
    elif i==2:
 #       create_and_save_bar_plot(histogram_data[:,0], histogram_data[:,1], column_labels[i], titles[i], filename, 8, 6.5, 13, 0.225, i)
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 8, 6.5, 13, 0.225, i)
    elif i==3:
 #       create_and_save_bar_plot(histogram_data[:,0], histogram_data[:,1], column_labels[i], titles[i], filename, 5.5, 8, 16, 0.2, i)
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 5.5, 8, 16, 0.2, i)
    elif i ==4:
 #       create_and_save_bar_plot(histogram_data[:,0], histogram_data[:,1], column_labels[i], titles[i], filename, 1, 8, 16, 0.2, i)
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 1, 8, 16, 0.2, i)
    elif i ==5:
 #       create_and_save_bar_plot(histogram_data[:,0], histogram_data[:,1], column_labels[i], titles[i], filename, 1, 8, 16, 0.15, i)
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 1, 8, 16, 0.15, i)