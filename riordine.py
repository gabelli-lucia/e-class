import csv

def merge_and_move_rows(first_csv_file, second_csv_file, id):
    # Read data from the first CSV file
    with open(first_csv_file, 'r', encoding='utf-8-sig') as file:
        first_reader = csv.reader(file)
        first_rows = list(first_reader)
        first_header = first_rows[0]

    # Read data from the second CSV file
    with open(second_csv_file, 'r', encoding='utf-8-sig') as file:
        second_reader = csv.reader(file)
        second_rows = list(second_reader)

    # Sistema il fatto che ci siano meno domande equity per il caso inglese
    if id=='PRE':
        equity = [i for i, col in enumerate(second_rows[0]) if 'Q44_Equity->None of the above' in col]
    elif id=='POST':
        equity = [i for i, col in enumerate(second_rows[0]) if 'Q46_Equity->None of the above' in col]
        errore = [i for i, col in enumerate(second_rows[0]) if 'Q41_30->What do YOU think while doing experiments for class?' in col]

    for row in second_rows:
        row.insert(equity[0]+1,'')
        row.insert(equity[0],'')
        row.insert(equity[0],'')
    if id=='POST':
        for row in second_rows:
            row.insert(errore[0],'')


	# Merge datasets
    merged_rows = first_rows[1:] + second_rows[1:]  # Ignore headers

    # Find columns with 'CHECK' in their header
    check_columns = [i for i, col in enumerate(first_header) if 'CHECK' in col]

    # Move rows as before
    if check_columns:
        for col in check_columns:
            for row in merged_rows:
                if int(row[col]) != 4:  # Adjust index to account for added "source" column
                    merged_rows.append(merged_rows.pop(merged_rows.index(row)))

    # Add new columns (domande post-only)
    off = 23	
	
    new_header = first_header + ["S->Risposte numeriche aggregate"]        
    for i in range(1, 23):                                              
        new_header.append(f"{i:02d} YOU")
        new_header.append(f"{i:02d} EX")

    new_header.append(f"CHECK_1")
    new_header.append(f"CHECK_2")

    for i in range(23, 31):
        new_header.append(f"{i:02d} YOU")
        new_header.append(f"{i:02d} EX")

    if id == 'POST':
        for i in range(1, off + 1):
            new_header.append(f"31_{i}")
	
    for row in merged_rows:
        if id == 'PRE':
            row.extend([""] * 63)
        else:
            row.extend([""] * (63 + off))

    you_ex_columns = [i for i, col in enumerate(first_header) if any(stringa in col for stringa in ['Cosa pensi TU mentre svolgi gli esperimenti durante un corso?', 'Come risponderebbero i fisici sperimentali riguardo le loro ricerche?'])]
    exp_columns = [i for i, col in enumerate(first_header) if 'Quanto era importante, per ottenere un buon voto in questo corso,' in col]

    # Copy values
    for row in merged_rows:
        if id == 'PRE':
            for i in range(len(you_ex_columns)):
                row[i - 62] = row[you_ex_columns[i]]
        else:
            for i in range(len(you_ex_columns)):
                row[i - 62 - off] = row[you_ex_columns[i]]
            for i in range(len(exp_columns)):
                row[i - off] = row[exp_columns[i]]
				
    # Write merged and modified data back to the first CSV file
    with open(id+'.csv', 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_header)
        writer.writerows(merged_rows)

if __name__ == "__main__":
    merge_and_move_rows('ECLASS_Ita_PRE_2S_.csv', 'ECLASS_Eng_PRE_2S_.csv', 'PRE') # Prima quello italiano, poi quello inglese
    merge_and_move_rows('ECLASS_Ita_POST_2S_.csv', 'ECLASS_Eng_POST_2S_.csv', 'POST') # Prima quello italiano, poi quello inglese
    print("Rows moved successfully.")
