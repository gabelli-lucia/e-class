import csv

def import_data(file_path, columns_to_keep):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        selected_columns_indices = [headers.index(col) for col in columns_to_keep if col in headers]
        
        for row in reader:
            filtered_row = list(row[idx] for idx in selected_columns_indices)
            # Cleaning up the fifth column (third student question)
            fifth_column_index = 4  # assuming fifth column index is always present
            fifth_column_value = filtered_row[fifth_column_index]
            # Rule 1: if value can be converted to an integer and greater than 31, set to ''
            try:
                if int(fifth_column_value) > 31:
                    filtered_row[fifth_column_index] = ''
            except ValueError:
                pass  # if conversion to int fails, just leave the value as it is
            
            # Rule 2: If length of value is greater than 2, keep just the first two characters
            if len(fifth_column_value) > 2:
                filtered_row[fifth_column_index] = fifth_column_value[:2]
            
            data.append(tuple(filtered_row))
    return data

def piano(tup1, tup2):
    # Split strings by semicolons and commas, then count occurrences of each word
    str1 = tup1[0]
    str2 = tup2[0]
    words_count1 = {}
    for word in str1.split(';'):
        words_count1[word] = words_count1.get(word, 0) + 1
    
    words_count2 = {}
    for word in str2.split(';'):
        words_count2[word] = words_count2.get(word, 0) + 1

    # Check if both dictionaries are equal
    return words_count1 == words_count2

def canale(tup1, tup2):
    str1 = tup1[-2]
    str2 = tup2[-2]
    val = str1==str2 or \
       ('1 : Canale 1/A/A-L/matr.0-4' in (str1, str2) and '1 : Track 1/A/A-L/ID0-4' in (str1, str2)) or \
       ('2 : Canale 2/B/M-Z/matr.5-9' in (str1, str2) and '1 : Track 2/B/M-Z/ID5-9' in (str1, str2)) or \
       ('3 : Canale 3/C' in (str1, str2) and '4 : Track 3/C' in (str1, str2)) or \
       ('4 : Canale 4/D' in (str1, str2) and '4 : Track 4/D' in (str1, str2)) or \
       ('5 : Canale 5/E' in (str1, str2) and '5 : Track 5/E' in (str1, str2)) or \
       ('6 : Senza canali' in (str1, str2) and '6 : No tracks' in (str1, str2))
    return val

def matricola(tup1, tup2):
    str1 = tup1[-3]
    str2 = tup2[-3]
    val = (str1==str2 or str1=='' or str2=='')
    return val

def find_matching_rows(data1, data2):
    matching_indices = []
    for i, row1 in enumerate(data1):
        for j, row2 in enumerate(data2):
            # Check for matching elements except for the first one (study plan) and the last three (matricola, channel and semester)
            if all(cell1.lower() == cell2.lower() for cell1, cell2 in zip(row1[1:-3], row2[1:-3])):
                # Check the last element for the special matching conditions
                if row1[-1].lower() == row2[-1].lower() or \
                   ('1 : primo anno' in (row1[-1].lower(), row2[-1].lower()) and '1 : first year - msc degree' in (row1[-1].lower(), row2[-1].lower())) or \
                   ('2 : secondo anno' in (row1[-1].lower(), row2[-1].lower()) and '2 : second year - msc degree' in (row1[-1].lower(), row2[-1].lower())):
                    matching_indices.append((i, j))
    return matching_indices

def filter_tuples(unsorted_lista, data1, data2, condition, iteration):
    # Check if there are doubles in the first elements
    lista = sorted(unsorted_lista, key=lambda x: x[0])
    print(f"Pre filtro abbiamo {len(lista)} corrispondenze.")
    i = 0
    while i < len(lista) - 1:
        if lista[i][0] == lista[i+1][0]:
            if (condition(data1[lista[i][0]], data2[lista[i][1]]) and condition(data1[lista[i+1][0]], data2[lista[i+1][1]]) and iteration==3) or \
               not (condition(data1[lista[i][0]], data2[lista[i][1]]) or condition(data1[lista[i+1][0]], data2[lista[i+1][1]])):
                del lista[i]
                del lista[i]
                continue
            elif condition(data1[lista[i][0]], data2[lista[i][1]]) and (not condition(data1[lista[i+1][0]], data2[lista[i+1][1]])):
                del lista[i+1]
            elif (not condition(data1[lista[i][0]], data2[lista[i][1]])) and condition(data1[lista[i+1][0]], data2[lista[i+1][1]]):
                del lista[i]
                continue
        i += 1

    # Check if there are doubles in the second elements
    lista = sorted(lista, key=lambda x: x[1])
    print(f"Post primo filtro abbiamo {len(lista)} corrispondenze.")
    j = 0
    while j < len(lista) - 1:
        if lista[j][1] == lista[j+1][1]:
            if (condition(data1[lista[j][0]], data2[lista[j][1]]) and condition(data1[lista[j+1][0]], data2[lista[j+1][1]]) and iteration==3) or \
               not (condition(data1[lista[j][0]], data2[lista[j][1]]) or condition(data1[lista[j+1][0]], data2[lista[j+1][1]])):
                del lista[j]
                del lista[j]
                continue
            elif condition(data1[lista[j][0]], data2[lista[j][1]]) and (not condition(data1[lista[j+1][0]], data2[lista[j+1][1]])):
                del lista[j+1]
            elif (not condition(data1[lista[j][0]], data2[lista[j][1]])) and condition(data1[lista[j+1][0]], data2[lista[j+1][1]]):
                del lista[j]
                continue
        j += 1
    print(f"Post secondo filtro abbiamo {len(lista)} corrispondenze.")

    return lista
		
		
def save_matched_indices(matched_rows, offset, file_path):
    with open(file_path, 'w') as file:
        for i in range(len(matched_rows)):
            # Shift indices when writing to the file
            file.write(f"Match at post index {matched_rows[i][1] + 1} and pre index {matched_rows[i][0] + 1 + offset}\n")

# Columns to keep for PRE data
columns_to_keep = ["Piano di studi", "Codice Corso di Laurea", "Q01_ID-1", "Q02_ID-2", "Q03_ID-3", "Q04_Matricola", "Q08_Course-2", "Q09_Course-3"]   ###Queste andranno modificate

# Import data
pre_data = import_data('PRE.csv', columns_to_keep)
post_data = import_data('POST.csv', columns_to_keep)

# Find matching indices
matching_indices = find_matching_rows(pre_data, post_data)

# Remove the doubles by checking the matricola
filtered_indices_by_matricola = filter_tuples(matching_indices, pre_data, post_data, matricola, 1)

# Remove the doubles by checking channels
filtered_indices_by_channel = filter_tuples(filtered_indices_by_matricola, pre_data, post_data, canale, 2)

# Remove the doubles by checking study plans
filtered_indices_by_plan = filter_tuples(filtered_indices_by_channel, pre_data, post_data, piano, 3)

# Write the indices to a file
save_matched_indices(filtered_indices_by_plan, len(post_data), 'Matched indices.txt')