import torch

## Esempio: suppongo di avere un tensore che rappresenta le uscite di un layer di una RN,
##  o valori di un certo Datast.
## Vedo ora come applicare diverse operazioni di riduzione per estrarre la somma, la media, e i valori
## massimi e minimi

tensore_a = torch.tensor( [1, 2, 3, 4, 5], dtype=torch.float32 )

# Somma di tutti gli elementi del tensore
risultato_somma = torch.sum( tensore_a ) 
print("Somma degli elementi del tensore: ", risultato_somma)

# Media di tutti gli elementi del tensore
risultato_media = torch.mean( tensore_a ) #i dati devono essere in formato float!!
print("\nMedia degli elementi del tensore: ", risultato_media)

# Calcolo del valore massimo e della sua posizione
max_val, max_idx = torch.max(tensore_a, dim=0) # calcolo il massimo lungo la prima dimensione del tensore (dim=0)
print("\nValore massimo: ", max_val)
print("Posizione del valore massimo: ", max_idx)


# Calcolo del valore minimo e della sua posizione
min_val, min_idx = torch.max(tensore_a, dim=0)
print("\nValore minimo: ", min_val)
print("Posizione del valore minimo: ", min_idx)

# Le posizioni del valore massimo o del valore minimo le ottengo anche usando argmax e argmin
max_position = torch.argmax(tensore_a)
min_position = torch.argmin(tensore_a)
print("\nPosizione del valore massimo con argmax: ", max_position)
print("Posizione del valore minimo con argmax: ", min_position)


## torch.max() può essere utile per identificare il vincitore in scenari come la classificazione.
## torch.argmin() possono essere utili per il tracciamento e l'analisi di posizioni critiche, senza dover gestire i valori effettivi.


###########################################
## Posso specificare su quali dimensioni del tensore applicare la riduzione.
matrice = torch.randn(3, 4)

print("\nGenero la seguente matrice: ", matrice)

# Calcolo la media su ogni riga
media_righe = torch.mean(matrice, dim=1)
print("Calcolo la media su ogni riga: ", media_righe)


#############################################
## Altre operazioni:
# - torch.var   calcola la varianza di un tensore
# - torch.std   calcola la deviazione standard di un tensore
# - torch.prod  calcola il prodotto di tutti gli elementi di un tensore
# - toch.all    restituisce True se tutti gli elementi di un tensore sono True 