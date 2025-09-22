import torch

# Definisco due tensori: il primo di dim. 3x1, il secondo di dim 1x3
tensore_1 = torch.ones(3, 1) # 3x1
tensore_2 = torch.ones(1, 3) # 1x3

# Li sommo:
somma = tensore_1 + tensore_2
print(somma)

## Il risultato è un tensore 3x3:
#  Infatti, in questo caso il tensore_2 viene automaticamente espanso a 3x3 per allinearsi con il tensore_1.
# La somma viene quindi eseguita elemento per elemento