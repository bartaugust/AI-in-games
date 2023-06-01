# Game objects generation

## Przygotowanie do uruchomienia

Wymagane biblioteki zawarte w pliku requirements.txt.

'''
pip install -r requirements.txt
'''

Konfiguracja projektu w folderze conf.
Przy pierwszym uruchomieniu należy ustawić ścieżki wpliku paths.yaml.

wrk - ścieżka projektu
datasets - lokalizacja zbiorów danych
output - folder, w którym zapisywane są logi
generated - folder, w którym zapisywane są wygenerowane obrazy

Dla domyślnej konfiguracji wystarczy zmienić ścieżkę do wrk.

## Dodawanie zbiorów danych
W celu dodania nowego zbioru danych należy utworzyć plik yaml w folderze conf/dataset.
W pliku zawarte są:
name - nazwa modelu,
load - funkcja wczytująca,
transform - użyte transformacje na obrazach
data_loader - parametry dla pytorch dataloader

W przypadku użycia funkcji ImaageFolder dla load nalezy utworzyć folder o nazwie podanej w parametrze root z dodatkowym subfolderem train zawierający obrazy.
