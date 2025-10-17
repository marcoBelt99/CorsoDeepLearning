'''
Autograd e calcolo del gradiente
Obiettivo: Definire una funzione semplice e calcolare i gradienti rispetto a variabili scalari
Definisci due variabili x e y come tensori scalari con ‘require_grad=True’
Definisci una funzione 𝑓(𝑥,𝑦)=3𝑥^2+2𝑦^2+𝑥𝑦
Calcola il gradiente di 𝑓 rispetto a 𝑥 e 𝑦
'''
import torch

# Definisco due variabili scalari con requires_grad = True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Definisco la funzione
f = 3 * x**2 + 2 * y**2 + x * y

# Calcolo i gradienti
f.backward()

# Stampo i gradienti
print('Gradiente rispetto a x: ', x.grad)
print('Gradiente rispetto a y: ', y.grad)
