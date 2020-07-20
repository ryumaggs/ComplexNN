import tensorflow as tf

def Load(path):
	'''

	'''
	f_in = open(path, 'r')
	
	line = f_in.readline()
	x_real = []
	x_imag = []
	y_real = []
	y_imag = []
	while(line != ''):
		line = line[:-1]
		line = line.split(',')
		x_real.append(float(line[0]))
		x_imag.append(float(line[1]))
		y_real.append(float(line[2]))
		y_imag.append(float(line[3]))
		line = f_in.readline()
		#print('a', line != '')
	tensor_x = tf.complex(x_real, x_imag)
	tensor_y = tf.complex(y_real, y_imag)

	return tensor_x, tensor_y

