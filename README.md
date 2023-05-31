# Klasyfikacja tekstów przy użyciu stylometrii (wektorów cech lingwistycznych)

### Instalacja
Instrukcja instalacji narzędzia Stylometrix znajduje się na [stronie projektu](https://github.com/ZILiAT-NASK/StyloMetrix).

### Uruchomienie
Aby uruchomić narzędzie z użyciem modelu Stylometrix na zbiorze AG News
```
python main.py -m stylo --train-data data_ag_news_train.csv --test-data data_ag_news_test.csv
```
Dodatkowe parametry wywołania:
```
-m, --model - model do klasyfikacji - stylo, bert
-s, --save-model - informacja czy zapisać wynikowy model do pliku
-o, --output-name - nazwa wynikowego pliku z modelem
-f, --from-file - wczytanie wytrenowanego wcześniej modelu z pliku
-d, --data - plik z wszystkimi całym zbiorem danych w formacie csv (obowiązkowy gdy nie ma podanych plików dla zbioru treningowego i testowego)
--train-data - plik z danymi treningowymi w formacie csv
--test-data - plik z danymi testowymi w formacie csv
--train-data-label - etykieta po jakiej powinien zostać podzielony plik z połączonymi danymi
--test-data-label - j.w.
--train-data-count - liczba przykładów jakie powinny być użyte do trenowania
--test-data-count - liczba przykładów jakie powinny być użyte do predykcji
--text-column - nazwa kolumny z tekstem
--label-column - nazwa kolumny z etykietą
-g, --greedy - informacja czy ma zostać przeprowadzony zachłanne trenowanie, które nie przewiduje błędnych danych
```