# Uso

Questo script elabora file di PRE, MID e POST di E-CLASS.

```
usage: do.py [-h] [--threshold THRESHOLD] [--matricola MATRICOLA] [--lang LANG] input_files [input_files ...]

Process E-CLASS survey data.

positional arguments:
  input_files           Input PRE file(s). MID and POST files will be derived automatically.

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Effect size threshold (default: 0.05)
  --matricola MATRICOLA
                        Matricola column name (default: Matricola)
  --lang LANG           File language (it, en; default: it)

```

Nota: Per ogni file PRE specificato, lo script cercherà automaticamente i corrispondenti file MID e POST sostituendo "_PRE_" con "_MID_" e "_POST_" nei nomi dei file.

# Compilare in un eseguibile

Usare pyinstaller:

`pip install -U pyinstaller`

`pyinstaller --onefile do.py --distpath . -n e-class`

# Deprecato

## Istruzioni per la versione originale

Usare in ordine questi script:

* `riordine.py` prende il file originale di Moodle e produce `PRE.csv` e `POST.csv`.
* `find_match.py` trova le corrispondenze tra `PRE.csv` e `POST.csv` e crea un file di testo `Matched indices.txt` con gli abbinamenti.
* A questo punto abbiamo `PRE.csv`, `POST.csv` e `Matched indices.txt`.

* In qualche modo produciamo il file `Analisi dati.csv`.

* Si esegue `annota_corrispondenze.py`.
