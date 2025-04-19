# Compilare in un eseguibile

Usare pyinstaller:

`pip install -U pyinstaller`
`pyinstaller --onefile do.py`

# Deprecato

## Istruzioni per la versione originale

Usare in ordine questi script:

* `riordine.py` prende il file originale di Moodle e produce `PRE.csv` e `POST.csv`.
* `find_match.py` trova le corrispondenze tra `PRE.csv` e `POST.csv` e crea un file di testo `Matched indices.txt` con gli abbinamenti.
* A questo punto abbiamo `PRE.csv`, `POST.csv` e `Matched indices.txt`.

* In qualche modo produciamo il file `Analisi dati.csv`.

* Si esegue `annota_corrispondenze.py`.

