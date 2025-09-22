# Il seguente script illustra come creare, addestrare e valutare un modello semplice di regressione,
# usando il dataset iris, che contiene: 150 dati (I fiori di iris) presenti nel dataset sono divisi in 3 specie.
# Ogni fiore è descritto da 4 caratteristiche: lunghezza e larghezza del sepalo, e lunghezza e larghezza del petalo

from sklearn.datasets import load_iris # import la funzione per caricare il DS iris
from sklearn.model_selection import train_test_split # funzione per dividere il DS in set di addestramento e di test
from sklearn.linear_model import LogisticRegression # importo il modello di regressione logistica, che userò per classificare le specie di iris

# carico il DS nella variabile iris
iris = load_iris()

# estraggo le caratteristiche
X = iris.data

# estraggo le etichette
y = iris.target

# divido il DS in 80% di addestramento, e 20% set di test. 
# Uso random state per garantire che la suddivisione sia riproducibile. 
# Questo passaggio è importante per testare l'efficacia del modello su dati non visti durante l'addestramento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creo l'istanza del modello di regressione logistica
model = LogisticRegression()

# addestro il modello sui dati di addestramento: durante questo processo, il modello impara ad associare le features (X_train)
# alle etichette corrispondenti (labels, ossia y_train)
model.fit(X_train, y_train)

# valutazione del modello: valuto l'accuratezza del modello sui dati di test.
score = model.score(X_test, y_test)
print(f"Accuratezza del modello: {score:.2f}")

# ora ho accuracy al 100% (1.00). Questo perchè il DS usato (iris) è semplice e ben separato.
# Questo significa che i modelli di ML, spesso raggiungono alte prestazioni. 
# Dei risultati simili potrebbero non essere replicabili su DS più complessi e meno distintamente separati.
# Poi, anche se non è solito ottenere un'alta accuracy col DS iris, un 100% può suggerire spesso un rischio di overfitting,
# specialmente se il DS di test è piccolo.
# Overfitting = quando il modello impara a memorizzare i dati di addestramento, incluse le anomalie, e quindi non riesce a generalizzare
# su questi dati.
# Però, appunto, visto che il DS è ben capito e spesso utilizzato per scopi di didattici (come in questo caso), il risultato
# è un segno della bontà del modello (per il DS specifico).
# accuracy del 100% ci dice che il modello di regressione logistica, per la classificazione delle specie di iris dimostra che le caratteristiche
# selezionate sono molto indicative delle diverse specie di iris.