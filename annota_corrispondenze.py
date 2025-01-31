import csv

# Function to extract X and Y indices from the string
def extract_indices(string):
    indices = string.split(' ')
    return int(indices[4]), int(indices[-1])

# Read the txt file and extract X, Y pairings
pairings = []
with open('Matched indices.txt', 'r') as file:
    for line in file:
        pairings.append(extract_indices(line.strip()))

# Read the csv file
with open('Analisi dati.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Identify the relevant columns
columns_match = [i for i, col in enumerate(rows[0]) if (col == 'Matching' or col == 'Riferimento Riga' or col =='Uomo-Donna (Parziale)' or col == 'Uomo-Donna')]

# Change the relevant columns at the rows specified by x, y
for couple in pairings:
    rows[couple[0]][columns_match[0]] = 'Matched'
    rows[couple[0]][columns_match[1]] = couple[1]
    rows[couple[1]][columns_match[0]] = 'Matched'
    rows[couple[1]][columns_match[1]] = couple[0]
    rows[couple[1]][columns_match[2]] = rows[couple[0]][columns_match[3]]

# Identify the relevant columns
columns_source = [i for i, col in enumerate(rows[0]) if (col == 'source' or col == 'PRE-POST')]
# Change all the rows at the specified column                        
for row in rows[1:]:
    row[columns_source[1]]=row[columns_source[0]]


# Delete the first 5 columns from each row (4 filtri + 1 source)
for row in rows:
    del row[:5]

with open('Analisi dati.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
