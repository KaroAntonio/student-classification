import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import confusion_matrix
from data_util.utils import get_dataset_name

class BatchLoader:
	def __init__(self, train_set, test_set, batch_size, verbose=False):
		self.batch_size = batch_size
		self.train_ptr, self.test_ptr = [-1], [-1]

		self.train_x = self.batchify(train_set['x'], batch_size)
		self.train_y = self.batchify(train_set['y'], batch_size)
		self.test_x = self.batchify(test_set['x'], batch_size)
		self.test_y = self.batchify(test_set['y'], batch_size)

		self.n_train = len(self.train_x)//2
		self.n_test = len(self.test_x)//2
		self.num_batches = self.n_train + self.n_test
		
		if verbose:
			print("n train batches: {}, n test batches: {}".format(self.n_train, self.n_test))
			print("feature dimensionality: {}".format(np.array(self.train_x).shape[-1]))

	def batchify(self, data, batch_size):
		'''
		return (data*2).split_into_batches()
		'''
		double_data = np.array(data*2)
		num_batches = len(double_data)//batch_size
		abridged_data = np.array(double_data[:num_batches*batch_size])
		batches = np.split(abridged_data, num_batches)
		return batches
		

	def next_batch(self, ptr, x, y):
		ptr[0] = (ptr[0] + 1) % len(x)
		return x[ptr[0]] , y[ptr[0]]

	def next_train(self):
		return self.next_batch(self.train_ptr, self.train_x, self.train_y)

	def next_test(self):
		return self.next_batch(self.test_ptr, self.test_x, self.test_y)

def build_confusion_matrix(bl, pred_max, y_max, x, y, keep_prob):
	'''
	bl: BatchLoader instance
	pred_max, y_max: tf ops that give the arg max for the respective ys
	return the confusion matrix for testing on a round of batches
	'''
	y_true = []
	y_pred = []
	for i in range(bl.n_test):
		batch_x, batch_y = bl.next_test()
		y_pred += list(pred_max.eval({
						x: batch_x, 
						y: batch_y, 
						keep_prob: 1.}))
		y_true += list(y_max.eval({
						x: batch_x, 
						y: batch_y, 
						keep_prob: 1.}))

	return confusion_matrix(y_true,y_pred)

def display_confusion_matrix( cm ):
	'''
	a confusion matrix of size nxm 
	display it appropriately
	'''
	print("Confusion Matrix")	
	print( cm )


def train( verbose = False, dataset_name = '201601' ):
	#LOAD
	with open('out/xy_{}_train.json'.format(dataset_name), 'r') as f:
		train_set = json.load(f)

	with open('out/xy_{}_test.json'.format(dataset_name), 'r') as f:
		test_set = json.load(f)

	# MODEL
	learning_rate = 0.001
	training_epochs = 50
	display_step = 5
	batch_size = 4
	dropout_rate = 0.75

	# PREP BATCHES
	bl = BatchLoader(train_set, test_set, batch_size, verbose)

	# Network Parameters
	n_hidden_1 = 60 # 1st layer number of features
	n_hidden_2 = 60 # 2nd layer number of features
	n_input = len(train_set['x'][0]) 
	n_classes = len(train_set['y'][0]) 
	
	# tf Graph input
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

	# Create model
	def multilayer_perceptron(x, weights, biases):
		# Hidden layer with RELU activation
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.relu(layer_1)
		# Hidden layer with RELU activation
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.relu(layer_2)
		# Hidden layer with RELU activation
		layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
		layer_3 = tf.nn.relu(layer_3)
		# Output layer with linear activation
		out_layer = tf.add(tf.matmul(x, weights['out']) , biases['out'])
		return out_layer

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.ones([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2])),
		'h3': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.ones([n_input, n_classes]))
	}

	biases = {
		'b1': tf.Variable(tf.ones([n_hidden_1])),
		'b2': tf.Variable(tf.zeros([n_hidden_2])),
		'b3': tf.Variable(tf.zeros([n_hidden_2])),
		'out': tf.Variable(tf.ones([n_classes]))
	}

	# Construct model
	pred = multilayer_perceptron(x, weights, biases)
	
	# Dropout
	#pred = tf.nn.dropout(pred, keep_prob);

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initializing the variables
	init = tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0.
			# Loop over all batches
			for i in range(bl.num_batches):
				batch_x, batch_y = bl.next_train()
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
															  y: batch_y,
															  keep_prob: dropout_rate})
				# Compute average loss
				avg_cost += c / bl.num_batches
			# Display logs per epoch step
			if epoch % display_step == 0:
				# Test model
				correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
				# Calculate accuracy
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				batch_x, batch_y = bl.next_test()
				acc = accuracy.eval({x: batch_x, y: batch_y, keep_prob: 1.})

				if verbose:
					print "Epoch:", '%04d' % (epoch+1), "cost: ", \
						"{:.9f}, accuracy: {:.9f}".format(avg_cost, acc)

		if verbose:
			print "Optimization Finished!"
		
		total_accuracy = 0
		pred_max = tf.argmax(pred, 1)
		y_max = tf.argmax(y, 1)
		correct_prediction = tf.equal(pred_max, y_max)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		
		# CONFUSION MATRIX
		cm = build_confusion_matrix( bl, pred_max, y_max, x, y, keep_prob)
		if verbose:
			display_confusion_matrix( cm )

		# VALIDATE Accuracy
		for i in range(bl.n_test):
			batch_x, batch_y = bl.next_test()
			total_accuracy += accuracy.eval({
											x: batch_x, 
											y: batch_y, 
											keep_prob: 1.})
		final_acc = total_accuracy / bl.n_test

		return final_acc, cm


if __name__ == "__main__":

	dataset_name = get_dataset_name()	
	final_acc, cm  = train(True, dataset_name)
	print("Final Accuracy: {}".format(final_acc) )

