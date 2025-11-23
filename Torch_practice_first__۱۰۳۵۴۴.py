# import torch
import torch as tr

scaler = tr.Tensor([1])
victor = tr.Tensor([1,2,3,5])
matrix = tr.tensor([[1,2,3,4,5],[6,7,8,9,0]])
tensor_ = tr.tensor([[[12,32,12,23,43],
                      [21,32,34,45,64],
                      [90,2,1,2,3],
                      [12,22,11,22,32]]])
print(scaler,"\nVector=>",victor,"\nMatrix=>",matrix,"\nTensor=>\n",tensor_,"\n")
mult = tensor_*scaler
print(tensor_, "*" ,scaler, " = \n",mult)
random_tensor = tr.rand(100,100)
print("Random_Tensor \n ",random_tensor,"\n",random_tensor.shape,"\n", random_tensor.ndim)
# Create a Zero and one Random number
ones = tr.ones(size = (100,100))
print("Ones \n ",ones,"\n",ones.shape,"\n",ones.ndim)
zeros = tr.zeros(size = (120,120))
print("Zeros \n ",zeros,"\n",zeros.shape,"\n",zeros.ndim)
# zeros multi with matrix
multZeros = zeros*scaler

print(multZeros)
# create range in pytorch which exist zeros, one
one_to_end = tr.arange(1,20)
print("One_to_end_range \n ",one_to_end,"\n",one_to_end.shape)
ten_zeros = tr.zeros_like(input = one_to_end)
print("ten_zeros \n ",ten_zeros,"\n",ten_zeros.shape)