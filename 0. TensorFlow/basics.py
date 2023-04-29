import tensorflow as tf
from  misc_utility.section import section

section('Tensors a')
x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])
print(x)
print(x.shape)
print(type(x))
print(x.dtype)

section('Tensors b')
print(x + x)
print(5 * x)
print(x @ tf.transpose(x))
print(tf.concat([x, x, x], axis=0))
print(tf.nn.softmax(x, axis=-1))
print(tf.reduce_sum(x))
print(tf.convert_to_tensor([1,2,3]))
print(tf.reduce_sum([1,2,3]))

section('Variables')
if tf.config.list_physical_devices('GPU'):
    print('TensorFlow **IS** using the GPU')
else:
    print('TensorFlow **IS NOT** using the GPU')

var = tf.Variable([0.0, 0.0, 0.0])
var.assign([1,2,3])
print(var)
var.assign_add([1,1,1])
print(var)

section('Automatic differentiation')
x = tf.Variable(1.0)
def f(x):
    y = x**2 + 2*x - 5
    return y
print(f(x))
with tf.GradientTape() as tape:
    y = f(x)
g_x = tape.gradient(y, x) # g(x) = dy/dx
print(g_x)

section('Graphs and tf.function')
@tf.function
def my_func(x):
    print('Tracing.\n')
    return tf.reduce_sum(x)
x = tf.constant([1,2,3])
print(my_func(x))
x = tf.constant([10, 9, 8])
print(my_func(x))
x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
print(my_func(x))

section('Modules, layers, and models')
class MyModule(tf.Module):
    def __init__(self, value):
        self.weight = tf.Variable(value)
    
    @tf.function
    def multiply(self, x):
        return x * self.weight

mod = MyModule(3)
print(mod)
print(mod.multiply(tf.constant([1,2,3])))
save_path = './saved'
tf.saved_model.save(mod, save_path)
reloaded = tf.saved_model.load(save_path)
print(reloaded.multiply(tf.constant([1,2,3])))

section('Training loops')













section('End')