import torch as tr
# Random Tensor
Random_tensor = tr.rand(3,4)
print(Random_tensor)
print(type(Random_tensor))
print("Random_size: ",Random_tensor.size(),"\n The number of Dimenctional",Random_tensor.ndim,"\n Device: ",Random_tensor.device)
print("==================")

### Tensor Operation, which usser create the operation
# add
add_tensor_1 = tr.tensor([1,2,3,4,5])
print(add_tensor_1+20)
add_tensor_2 = tr.tensor([1,2,3,4,5])
n = 10
add_tensor_2_ = add_tensor_2 + n
print("Addition: ",add_tensor_2," + ",n," = ",add_tensor_2_)
# Multplication
mult_tensor_1 = tr.tensor([1,2,3,4])
mult_tensor_2 = n * mult_tensor_1
print("Mutiplication: ",mult_tensor_1,"*",n," = ",mult_tensor_2)

# Subtraction
sub_tensor_1 = tr.tensor([1,2,3,4])
sub_tensor_2 = sub_tensor_1 - n
print("Subtraction: ",sub_tensor_1,"-",n," = ",sub_tensor_2)

# Devision
dev_tensor_1 = tr.tensor([10,20,30,40,50])
dev_tensor_2 = dev_tensor_1 / n
print("Division: ",dev_tensor_1," / ",n," = ",dev_tensor_2)

## Also we have bult-in operations, we can add two numbers or matrix or one number by one matrix
# add
print('======= BULT_IN OPERATION ===========')
add_tensor_2_ = tr.add(add_tensor_1,add_tensor_2)
print("Addition: ",add_tensor_1,"+",add_tensor_2," = ",add_tensor_2_)
# Subtract
sub_tensor_2_ = tr.sub(sub_tensor_1,3)
print("Subtraction: ",sub_tensor_1,"-",n," = ",sub_tensor_2)
# Multiplication == >mul
mult_tensor_2_ = tr.mul(mult_tensor_1,mult_tensor_2)
print("Multiplaction: ",mult_tensor_1,"*",mult_tensor_2, " = ",mult_tensor_2_)

print('<========= MATRIX MULTIPLICATION ===========>')
f_tensor= tr.tensor([12,23,34,45])
print("First tensor: ",f_tensor)
matrix_multiplaction = tr.matmul(f_tensor,f_tensor) # if we want to find the total of element we use the matmul after the multiplication
print("Matrix multiplication: ",matrix_multiplaction)
simple_matmul = f_tensor * f_tensor
print("Simple matmul: ",simple_matmul)

print('<=========== MATRIX MULTIPLICATION BY @ ============> ')
tensor_ =tr.tensor([1,3,3,2])
print('Matrix Multiplacation By @ : ',tensor_ @ tensor_)

print(" -------- End Of our tonight codes --------")

