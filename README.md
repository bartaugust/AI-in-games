# Game objects generation

## Przygotowanie

Wymagane biblioteki zawarte w pliku requirements.txt.

```
pip install -r requirements.txt
```

Konfiguracja projektu w folderze conf.
Przy pierwszym uruchomieniu należy ustawić ścieżki wpliku paths.yaml.


- wrk - ścieżka projektu
- datasets - lokalizacja zbiorów danych
- output - folder, w którym zapisywane są logi
- generated - folder, w którym zapisywane są wygenerowane obrazy

Dla domyślnej konfiguracji wystarczy zmienić ścieżkę do wrk.

## Dodawanie zbiorów danych
W celu dodania nowego zbioru danych należy utworzyć plik yaml w folderze conf/dataset.
W pliku zawarte są:
- name - nazwa modelu,
- load - funkcja wczytująca,
- transform - użyte transformacje na obrazach
- data_loader - parametry dla pytorch dataloader

W przypadku użycia funkcji ImaageFolder dla load nalezy utworzyć folder o nazwie podanej w parametrze root z dodatkowym subfolderem train zawierający obrazy.

## Modele
Dostępnymi modelami są dgan (64x64) i dgan128 (128x128).
Aby zmienić parametry, takie jak krok uczenia czy liczba epok należy zmienić je w pliku yaml modelu znajdującym się w folderze conf/model

## Uruchomienie

W pliku config.yaml należy wybrać zbiór danych z przygotowanych w conf/dataset oraz model z przygotowanych w conf/model.
Możliwe jest uruchomienie jednocześnie kilku zbiorów danych i kilku modeli..
W tym celu należy zmienić parametr hydra.mode z RUN na MULTIRUN i w hydra.sweeper.params wybrać zbiory danych i modele (glob(*) aby wybrać wszystkie w folderze).
Program uruchamia się przez uruchomienie funkcji train w pliku src/training/train_model.py
