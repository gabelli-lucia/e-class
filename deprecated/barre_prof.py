import csv
import numpy as np
import matplotlib.pyplot as plt
import textwrap

id = 'FM'

# Define the mapping from strings to numeric values
string_to_value = {
    "Mai": 0,
    "Raramente": 1,
    "Qualche volta": 2,
    "Spesso": 3,
    "Sempre": 4
}

# Define mapping from id to rows
if id == 'IA':
    riga = 5
elif id == 'IE':
    riga = 2
elif id == 'FFY':
    riga = 0
elif id == 'FSY':
    riga = 1
elif id == 'FTY':
    riga = 4
elif id == 'FM':
    riga = 6

# Load data from CSV file
data = []
with open('barre_prof.txt', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

# Convert strings to numeric values
numeric_data = [[string_to_value[item] for item in row[3:]] for row in data[1:]]

# Split data into separate histograms
histograms = [
    np.array([int(valore) for valore in data[riga+1][:3]]),    # First three columns
    np.array(numeric_data)[riga, :3],    # Next three columns
    np.array(numeric_data)[riga, 3:9],   # Next six columns
    np.array(numeric_data)[riga, 9:16],  # Next seven columns
    np.array(numeric_data)[riga, 16:20], # Next four columns
    np.array(numeric_data)[riga, 20:]    # Last four columns
]

# Retrieve the column labels
column_labels = []
column_labels.append(list(data[0][:3]))
column_labels.append(list(data[0][3:6]))
column_labels.append(list(data[0][6:12]))
column_labels.append(list(data[0][12:19]))
column_labels.append(list(data[0][19:23]))
column_labels.append(list(data[0][23:]))

# Set the titles of the plot
titles = [r'Qual $\grave{\rm e}$ il peso dei seguenti macro-obiettivi'+'\n'+'nella strutturazione del laboratorio?', 'Tipo di indagine: Le/gli studenti...','Margine di azione: Le/gli studenti...','Sviluppo e uso di modelli: Le/gli studenti...', \
   'Analisi e visualizzazione dei dati: Le/gli studenti...','Comunicazione: Le/gli studenti...']

# Set the y-ticks for the first figure
first_y_ticks = [r"Non $\grave{\rm e}$ un"+'\n'+"obiettivo", "Meno\ndel 50%", "50%", r'$Pi\grave{\rm u}$'+'\n'+'del 50%', r"$\grave{\rm E}$ l'unico"+'\n'+"obiettivo"]

# Function to create and save bar plot
def create_and_save_bar_plot(data, x_labels, title, filename, mult, size, wrap, bott, index):
    plt.bar([mult*value for value in range(len(data))], data, width=mult*0.8)
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
    if index !=0:
        plt.yticks(list(string_to_value.values()), ['\n'.join(textwrap.wrap(label, width=9)) for label in list(string_to_value.keys())], fontsize=8)  # Set y-tick labels
    else:
        #plt.yticks(list(string_to_value.values()), ['\n'.join(textwrap.wrap(label, width=9)) for label in first_y_ticks], fontsize=8)
        plt.yticks(list(string_to_value.values()), first_y_ticks, fontsize=8)
    plt.subplots_adjust(bottom=bott)  # Adjust the bottom margin as needed
    plt.gca().set_ylim(0,4.2)
    plt.savefig(filename, dpi=400)
    plt.close()

# Create and save bar plot for each element of histograms
for i, histogram_data in enumerate(histograms):
    # Select corresponding column labels for x-axis
    filename = f"{id} - Prof {i+1}.png"
    if i==0:
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 1, 8, 16, 0.2, i)
    elif i==2:
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 5.5, 5.75, 19, 0.15, i)
    elif i==3:
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 5.5, 5.5, 16, 0.2, i)
    elif i==1 or i==4:
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 1, 8, 16, 0.2, i)
    elif i ==5:
        create_and_save_bar_plot(histogram_data, column_labels[i], titles[i], filename, 1, 8, 16, 0.15, i)