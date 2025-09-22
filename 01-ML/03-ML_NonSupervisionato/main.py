# Uso dell'algoritmo K-Means (non supervisionato, metodo di clustering) per raggruppare un insieme di dati generati casualmente in cluster.
# Poi visualizzo il risultato del clustering tramite uno scatter plot

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt # per creare visualizzazioni. In questo caso userò lo scatter plot

# Genero un array di 100 punti in 2 dimensioni.
# X sarà l'input dell'alg. K-Means
X = np.random.rand(100, 2) 

# Creo un'istanza dell'alg. K-Means, specificando che i dati devono essere suddivisi in 3 cluster.
# Usando random_state=0 serve a garantire che i risultati siano riproducibili, ossia che ogni volta
# che eseguo il codice l'inizializzazione casuale è la stessa. 
# Il metodo fit(X) addestra l'algoritmo sui dati X, cercando di raggrupparli in 3 cluster
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Uso predict(X) per assegnare ogni punto del dataset a uno dei 3 cluster identificati.
# Il risultato sarà un array di etichette che indica a quale cluseter appartiene ciascun punto.
labels = kmeans.predict(X)

# Con scatter() creo una scatterplot dei punti in X, colorando ciascun punto in base al cluster di appartenenza.
# cmap specifica la mappa di colori usata per distinguere i cluster
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.show() # visualizzo il cluster
