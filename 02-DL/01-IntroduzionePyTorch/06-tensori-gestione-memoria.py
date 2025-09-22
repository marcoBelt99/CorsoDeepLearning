import torch

## Operazione out-of-place: crea una nuova copia del tensore con i dati modificati
tensor = torch.tensor( [1, 2, 3] )
risultato_out_of_place = tensor.add(1)
print(risultato_out_of_place)

## Operazione in-place: modifica i dati direttamente nel tensore originale
tensor.add_(1) # per fare un'operazione in-place devo usare l'underscore. 

## Vedo se la GPU è disponibile, e se lo passo dall'uso della CPU all'uso della GPU:
is_gpu_disponibile : bool = torch.cuda.is_available()
print("E' disponibile l'uso della GPU in questo computer? ", is_gpu_disponibile)
if is_gpu_disponibile:
    tensor = tensor.to('cuda')
    