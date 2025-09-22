import torch

tensore_a = torch.tensor([1, 2, 3])
tensore_b = torch.tensor([4, 5, 6])


## SOMMA tra tensori (elemento per elemento)
add_result = tensore_a + tensore_b
print("SOMMA: ", add_result)


## SOTTRAZIONE tra tensori (elemento per elemento)
sub_result = tensore_a - tensore_b
print("\nDIFFERENZA: ", sub_result)

## MOLTIPLICAZIONE tra tensori (elemento per elemento)
mul_result = tensore_a * tensore_b
print("\nMOLTIPLICAZIONE: ", mul_result)

## DIVISIONE tra tensori (elemento per elemento)
div_result = tensore_a / tensore_b
print("\nDIVISIONE: ", div_result)