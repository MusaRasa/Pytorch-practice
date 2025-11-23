import torch as tr
################### tensor datatypes
# 1) float Dtype
first_float_dtype_64 = tr.tensor([1.3,2.2,3.1,4.4,5.10], dtype = float,
                              device = None,
                              requires_grad = False
                              )
print(first_float_dtype_64)
# Now we want to convert float64 to float32
first_float_dtype_32=first_float_dtype_64.type(tr.float32)
print(first_float_dtype_32)
mult = first_float_dtype_64*first_float_dtype_32
print(mult)

# Now we work to int64,int 32
int_dtype = tr.tensor([3,6,9,12,15] ,dtype = tr.int64)
int_dtype_32 = int_dtype.type(tr.int32)
print(int_dtype, "\n",int_dtype_32)


# How we can find out details about tensors like shape, data_type, device
some_tensor = tr.randn((3,4))
print(some_tensor)
print(f"the shape of tensor: {some_tensor.shape}")
print(f"the Datatype of tensor: {some_tensor.dtype}")
print("now we want to convert int32 to int64")
some_tensor_64 = some_tensor.type(tr.int64)
print(some_tensor_64)
print(f"the device of tensor: {some_tensor.device}")