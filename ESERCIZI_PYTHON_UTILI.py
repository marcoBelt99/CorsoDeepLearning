## (1) Data una lista di nomi di file:
#   - creare una seconda lista avente come elementi il numero staccato dall'estensione e dal punto.
#   - tale lista di interi deve essere ordinata

lista_nomi_files_txt = ['4.txt', '2.txt', '3.txt', '5.txt', '1.txt']

lista_numeri_files_txt = sorted( list ( map( lambda nf : int( nf.split('.' )[0] ), lista_nomi_files_txt ) ) )

print(f"LISTA NOMI FILE NUMERICI: ", lista_numeri_files_txt)

## (2) - Creo una seconda lista avente come nomi i nomi dei file delle immagini (che hanno estensione .jpg)
#         con gli stessi numeri che ci sono nella lista dei file di testo
#      - verifico che, l'i-esimo file txt e l'i-esimo file jpg siano gli stessi numeri
lista_nomi_files_jpg = ['4.jpg', '2.jpg', '3.jpg', '5.jpg', '1.jpg']
lista_numeri_files_jpg = sorted( list ( map( lambda nf : int( nf.split('.' )[0] ), lista_nomi_files_jpg ) ) )

stessi_numeri : bool = all( n_txt == n_jpg for n_txt, n_jpg in zip( lista_numeri_files_txt, lista_numeri_files_jpg ) )
print( "stessi numeri" if stessi_numeri else "ci sono numeri diversi" )