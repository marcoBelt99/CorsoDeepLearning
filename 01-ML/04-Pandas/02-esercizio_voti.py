import pandas as pd

voti = pd.read_csv('voti.csv')

# Verifico di che tipo sono le colonne
print( voti.dtypes )

# Prime 6 righe
print( voti.head(6) )

print('\n')
# Stampo ora un conciso sommario del dataframe
# tramite il metodo info()
# Questo è utile per verificare:
#   totale di record presenti
#   numero di eventuali valori non null per ciascuna colonna e il loro tipo
print( voti.info() )

# Calcolare: voto medio, voto massimo, voto minimo per ciascuna materia
media, massimo, minimo = voti["voto"].mean(), voti["voto"].max(), voti["voto"].min()

print( 'Voto medio: ', f'{media:.2f}' )
print( 'Deviazione standard dei voti: ', f'{voti["voto"].std():.2f}' )
print( 'Voto massimo: ', f'{massimo}' )
print( 'Voto minimo: ', f'{minimo}' )


# Visualizzo gli indicatori statistici tramite il metodo describe(), il quale considera solo le colonne numeriche
print( voti.describe() )



# Creare un dataframe dei soli voti >= 25
voti_maggiori_uguali_25 = voti[voti["voto"] >= 25]

print('\nVoti maggiori di 25:')
print( voti_maggiori_uguali_25.head( len(voti_maggiori_uguali_25) ) )

# Creare il dataframe opposto al precedente (voti < 25)
print('\nVoti minori di 25') # posso fare una differenza insiemistica (riutilizzando quello che ho già trovato)
voti_minori_25 = voti[~voti["voto"].isin(voti_maggiori_uguali_25["voto"])] # notare l'uso dell'operatore tilde
print( voti_minori_25.head( len(voti_minori_25) ) )

# Salvo i dataframe in excel
voti_maggiori_uguali_25.to_excel('./voti_maggiori_uguali_25.xlsx')
voti_minori_25.to_excel('./voti_minori_25.xlsx')