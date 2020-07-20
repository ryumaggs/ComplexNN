import tensorflow as tf
import numpy as np
import Networks
import argparse
import DataLoader
parser = argparse.ArgumentParser(description='params for network and training')
parser.add_argument('--input_shape_', type = int, help='numbers seperated by space', nargs='+', default = [1])
parser.add_argument('--layers', help='list', type = int, nargs='+', default = [1])
parser.add_argument('--drop_out', type = int, help='dropout', default = 0)
parser.add_argument('--activation_func', type = str, default='relu', help='activation function, default is ReLU')
parser.add_argument('--training_data', help='filepath', type=str, default='./linear_complex_train.txt')
parser.add_argument('--testing_data', help='filepath', type=str, default='./linear_complex_test.txt')
def main(parser):
	args = parser.parse_args()
	args.input_shape_ = tuple(args.input_shape_)
	print(args)


	model = Networks.DeepNetwork(args)
	loss_func = tf.keras.losses.MeanSquaredError() 
	model.compile(optimizer = 'adam', loss = loss_func, metrics=[])
	#model.summary()

	train_x, train_y = DataLoader.Load(args.training_data)
	print(train_x.shape)
	model.fit(train_x, train_y, batch_size = train_x.shape[0],epochs = 1,steps_per_epoch=10)
	y_hat = model.predict(train_x,steps=1)
	true_y = np.asarray(train_y)

	i = 0

	print("prediction: ", y_hat)
	print("true label: ", np.asarray(true_y))

main(parser)
