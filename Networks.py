import tensorflow as tf
from tensorflow import keras
#import random
import numpy as np
#to print anything. use the following line:
#with tf.Session() as sess:  print(product.eval())
#where "product" is the tensor you want to print
def DeepNetwork(params):
	'''
	Params:
		input_shape_: <tuple>
		layers: <list> of nodes per layer (excludes input)
		activation_func: <string> name of act func
                drop_out: <int>
	'''
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(params.layers[0], input_shape=params.input_shape_))
	model.add(tf.keras.layers.Dropout(params.drop_out))
	i = 1
	while( i < (len(params.layers))):
		model.add(tf.keras.layers.Dense(params.layers[i], activation=params.activation_func))
		if(i != len(params.layers) -1):
			model.add(tf.keras.layers.Dropout(params.drop_out))
		i += 1
	return model

class ComplexDenseLayer(keras.layers.Layer):

	def __init__(self,input_dim,num_outputs):
		super(ComplexDenseLayer, self).__init__()
		self.num_outputs = num_outputs

		real = np.random.rand(input_dim,self.num_outputs)
		imag = np.random.rand(input_dim,self.num_outputs)
		complex_mat = tf.complex(real,imag)
		self.w = tf.Variable(complex_mat, dtype='complex128',trainable=True)
		bias_real = np.random.rand(self.num_outputs,1)
		bias_imag = np.random.rand(self.num_outputs,1)
		bias_complex = tf.complex(bias_real, bias_imag)
		self.b = tf.Variable(bias_complex,dtype='complex128',trainable=True)

	def call(self,inputs):
		batch_b = tf.repeat(self.b,inputs.shape[1],1)
		return tf.matmul(tf.transpose(self.w),inputs)+batch_b
'''
sess = tf.InteractiveSession()
real = tf.constant([1.0,2.0])
imag = tf.constant([3.0,4.0])
a = tf.complex(real,imag)
a = tf.reshape(a,(2,1))
b = tf.reshape(a,(1,2))
print("a: ", a.eval())
print("b:", b.eval())
output = tf.matmul(b,a)
#output = tf.Print(output, [output],message="output: ")
print(output.eval())
#print(out_print)
'''
x_real = [[1.0, 2.0],[3.0,4.0]]
x_imag = [[1.0,2.0],[3.0,4.0]]
x = tf.cast(tf.complex(x_real,x_imag),dtype='complex128')
print(x)
linear_layer = ComplexDenseLayer(2,4)
h1 = ComplexDenseLayer(4,2)
y = linear_layer(x)
print(y)
y = h1(y)
print(y)
