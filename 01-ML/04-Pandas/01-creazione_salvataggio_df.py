import pandas as pd

# Creo un dataframe e lo salvo su file
matematici = pd.DataFrame(
    {
        "Nome" : ['Alan Turing', 'John von Neumann', 'Marvin Lee Minsky', 'John Horton Conway'],
        "Anno di nascita" : [1952, 1903, 1927, 1937],
        "Anno di morte" : [1954, 1957, 2016, 2020]
    }
)

# Quali colonne sono presenti?
print("Sono presenti le seguenti colonne:")
print( matematici.columns )
print('\n')

# Accedo alla singola colonna
print( matematici['Nome'] )
print('\n')

# Stampo le prime 5 righe.
# Il metodo head() è utile richiamarlo dopo la lettura per verificare che:
# - siano stati riconosciuti le colonne presenti nel .csv e i loro nomi
# - il dataframe sia stato riempito correttamente con i dati
print( matematici.head() )

# Salvataggio dataframe su file
matematici.to_csv('./matematici_salvato.csv')
