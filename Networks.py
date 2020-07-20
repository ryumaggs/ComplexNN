import tensorflow as tf

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


