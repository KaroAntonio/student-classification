import tensorflow as tf
import numpy as np
import json

class BatchLoader:
	def __init__(self, x, y):
		self.batch_size = batch_size = 30 
		train_ratio = 0.9 # ration of batches dedicated to training
		self.train_ptr, self.test_ptr = [-1], [-1]

		self.num_batches = num_batches = len(x)//batch_size

		x = np.array(x[:num_batches*batch_size])
		y = np.array(y[:num_batches*batch_size])

		x_batches = np.split(x, num_batches)
		y_batches = np.split(y, num_batches)

		n_train = int(train_ratio * num_batches)

		self.train_x, self.test_x = x_batches[:n_train], x_batches[n_train:]
		self.train_y, self.test_y = y_batches[:n_train], y_batches[n_train:]

	def next_batch(self, ptr, x, y):
		ptr[0] = (ptr[0] + 1) % len(x)
		return x[ptr[0]] , y[ptr[0]]

	def next_train(self):
		return self.next_batch(self.train_ptr, self.train_x, self.train_y)

	def next_test(self):
		return self.next_batch(self.test_ptr, self.test_x, self.test_y)

#LOAD
with open('out/xy.json', 'r') as f:
	data = json.load(f)

bl = BatchLoader(data['x'], data['y'])

# MODEL
learning_rate = 0.001
training_epochs = 50
batch_size = bl.batch_size
display_step = 2

# Network Parameters
n_hidden_1 = 200 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features
n_input = len(data['x'][0]) # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

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
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

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
														  y: batch_y})
			# Compute average loss
			avg_cost += c / bl.num_batches
		# Display logs per epoch step
		if epoch % display_step == 0:
			# Test model
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			# Calculate accuracy
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			batch_x, batch_y = bl.next_test()
			acc = accuracy.eval({x: batch_x, y: batch_y})

			print "Epoch:", '%04d' % (epoch+1), "cost: ", \
					"{:.9f}, accuracy: {:.9f}".format(avg_cost, acc)

	print "Optimization Finished!"
	
	total_accuracy = 0
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	for i in range(len(bl.test_x)):
		batch_x, batch_y = bl.next_test()
		total_accuracy += accuracy.eval({x: batch_x, y: batch_y})
	print("Final Accuracy: {}".format(total_accuracy / len(bl.test_x)) )

