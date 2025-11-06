import os
import pandas as pd

DIR_ANNOTAZIONI = 'annotazioni'

num_totale_files = 13
lista_nomi_files_jpg = [ '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg',
                         '6.jpg', '7.jpg','8.jpg','9.jpg','11.jpg',
                         '12.jpg', '13.jpg','15.jpg'
                    ]

print(lista_nomi_files_jpg)


num_punti = 14


lista_nomi_files_txt = sorted( [nf for nf in os.listdir(DIR_ANNOTAZIONI)], key=lambda nf: int(nf.split('.')[0]) )
lista_nomi_files_jpg = sorted( lista_nomi_files_jpg, key=lambda nf: int(nf.split('.')[0]) )



dati : pd.DataFrame = pd.DataFrame( columns=['path_img'] + 
                             [f'punto_{n}_{coord}' for n in range(1, num_punti+1) for coord in ('X', 'Y')] ) # +1 per comprendere anche il 14°-esimo punto nel range


def prepara_dataset_completo(DIR_ANNOTAZIONI, lista_nomi_files_jpg, lista_nomi_files_txt, dati):
    for t, i in zip( lista_nomi_files_txt, lista_nomi_files_jpg ):
        try:
        ## La prima colonna deve avere il nome dell'immagine
            riga = {'path_img': i } # costruisco il dizionario per questa riga
        
        ## Le altre colonne, ciascuna deve avere la coordinata
            df_txt = pd.read_csv( f"{DIR_ANNOTAZIONI}/{t}", delimiter=',', header=0 ) # prima colonna è quella delle intestazioni
        
        ## Recupero le coordinate X ed Y rispettivamente
            X = df_txt['X'].values
            Y = df_txt['Y'].values

        ## Inizio col definirmi la variabile che rappresenta la singola riga del DF
        
            riga.update( {f'punto_{idx+1}_X' : x for idx, x in enumerate(X)} )
            riga.update( {f'punto_{idy+1}_Y' : y for idy, y in enumerate(Y)} )

       ## Aggiungo la riga al dataframe
            dati.loc[ len(dati) ] = riga
       
        
        except FileNotFoundError:
            print("Errore, file non esistente")
        except Exception as e:
            print(e)


prepara_dataset_completo(DIR_ANNOTAZIONI, lista_nomi_files_jpg, lista_nomi_files_txt, dati)



print()
print( dati )


RAGGRUPPAMENTI = {
    "GRUPPO1": [0, 1, 4, 5],    # S, N, A, B
    "GRUPPO2": [2, 3, 7, 8],    # Sna, Snp, Gn, Go
    "GRUPPO3": [9, 10, 11, 12], # U1r, U1t, L1r, L1t
    "GRUPPO4": [6, 13]          # Pg, Mesial
}


## Creo diversi DF ognuno correlato ad uno specifico gruppo di punti

def crea_sottodataframe_per_gruppo(dati_completi, nome_gruppo) -> pd.DataFrame:
    """
    Crea sotto-dataframe per il raggruppamento specificato"""
      
    # Seleziona tutte le colonne relative ai punti del gruppo (sia X che Y)
    colonne_selezionate = ['path_img']  # Manteniamo sempre il path dell'immagine
    
    for idx in RAGGRUPPAMENTI[nome_gruppo]:
        punto_numero = idx + 1  # Converto da indice 0-based a numero punto 1-based
        colonne_selezionate.extend([f'punto_{punto_numero}_X', f'punto_{punto_numero}_Y'])
    
    # Crea il sotto-dataframe, selezionando solo le colonne di interesse, e facendo una copia di quello originale
    dataframe_gruppo = dati_completi[colonne_selezionate].copy()
    
    return dataframe_gruppo




dati_gruppo_1 = crea_sottodataframe_per_gruppo(dati, "GRUPPO1")
dati_gruppo_2 = crea_sottodataframe_per_gruppo(dati, "GRUPPO2")
dati_gruppo_3 = crea_sottodataframe_per_gruppo(dati, "GRUPPO3")
dati_gruppo_4 = crea_sottodataframe_per_gruppo(dati, "GRUPPO4")


print( "\nGRUPPO 1\n" , dati_gruppo_1.head() )
print( "\nGRUPPO 2\n" , dati_gruppo_2.head() )
print( "\nGRUPPO 3\n" , dati_gruppo_3.head() )
print( "\nGRUPPO 4\n" , dati_gruppo_4.head() )

