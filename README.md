# Uso

Questo script elabora file di E-CLASS in due modalità:
1. Modalità PRE-POST (default): elabora file PRE e POST di E-CLASS
2. Modalità POST-POSTPOST: elabora file POST e POSTPOST di E-CLASS

Utilizzare l'argomento `--mode` per specificare la modalità desiderata.

```
usage: do.py [-h] [--threshold THRESHOLD] [--matricola MATRICOLA] [--lang LANG] [--mode {pre,post}] input_file

Process E-CLASS survey data.

positional arguments:
  input_file            Input file. In PRE-POST mode, this is the PRE file, and the POST files will be derived automatically.
                        In POST-POSTPOST mode, this is the POST file, and the POSTPOST files will be derived automatically.

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Effect size threshold (default: 0.05)
  --matricola MATRICOLA
                        Matricola column name (default: Matricola)
  --lang LANG           File language (it, en; default: it)
  --mode {pre,post}     Processing mode: pre (default) or post

```

# Compilare in un eseguibile

Usare pyinstaller:

`pip install -U pyinstaller`

`pyinstaller --onefile do.py --distpath . -n e-class`

# File di output

Lo script genera i seguenti file di output:

* `out-success.csv`: Contiene le percentuali di successo per ogni domanda
* `out-medie.csv`: Contiene le medie, i valori di Mann-Whitney e Cohen's d per ogni domanda
* `out-chart-means.png`: Grafico delle medie complessive per le risposte "YOU" e "Expert"
* `out-chart-what-do-you-think.png`: Grafico delle medie per le domande "What do YOU think"
* `out-chart-after-before.png`: Grafico comparativo dettagliato per tutte le domande

# Deprecato

## Istruzioni per la versione originale

I seguenti script sono deprecati e non dovrebbero essere utilizzati:

* `riordine.py` prende il file originale di Moodle e produce `PRE.csv` e `POST.csv`.
* `find_match.py` trova le corrispondenze tra `PRE.csv` e `POST.csv` e crea un file di testo `Matched indices.txt` con gli abbinamenti.
* `annota_corrispondenze.py`

Utilizzare invece lo script principale `do.py` come descritto sopra.
