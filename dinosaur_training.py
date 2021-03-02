'''
CHANGELOG

model():
	- add verbose
	- add print result after 2000 iteration (basicall the sample in the course)
'''

import numpy as np
from utils import *
import random
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

# Load data
data  = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
chars = sorted(chars)
ch_to_ix = {ch:i for i, ch in enumerate(chars)} 
ix_to_ch = {i:ch for i, ch in enumerate(chars)}


def sample(parameters, ch_to_ix, seed):
	'''
	Sampling a sequence of characters based on the probabilities outputed by a RNN

	Arguments:
		paramenters	-- dictionary containing weights of the RNN: Waa, Wax, Way
		ch_to_ix 	-- dictionary mapping index to characters
		seed 		-- for testing purpose
		
	Returns
		indices 	-- list of length n containing the indices of generated characters
	'''
	Waa = parameters['Waa']
	Wax = parameters['Wax']
	Wya = parameters['Wya']
	b = parameters['b']
	by = parameters['by']

	vocab_size = len(by)
	_, n_a = Waa.shape

	indices = []
	newline_character = ch_to_ix['\n']
	
	# initialise x and a
	x = np.zeros((vocab_size, 1))
	a_prev = np.zeros((n_a, 1))

	idx = -1
	counter = 0 

	# generate new word, loop until meet new line character or reach 50 chars
	while (idx != newline_character and counter != 50):
		# calc y
		a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
		y = softmax(np.dot(Wya, a) + by)

		# for testing purpose
		np.random.seed(counter + seed)

		# select idx from y (probabilities)
		idx = np.random.choice(range(len(y)), p=y.ravel())

		# one-hot encoding x
		x = np.zeros((vocab_size, 1))
		x[idx] = 1

		a_prev = a

		indices.append(idx)

		# increase counter
		counter += 1

	if counter == 50:
		indices.append(ch_to_ix['\n'])

	return indices

def optimize(X, Y, a_prev, parameters, learning_rate):
	'''
	optimize the parameters

	Arguments:
		X 	--
		Y 	--
		a_prev 		-- 
		parameters 	--
		learning_rate 	--

	Returns
		loss			--
		gradients		--
		a[len(X) - 1]	--
	'''
	
	# forwarding
	loss, cache = rnn_forward(X, Y, a_prev, parameters)

	# back probagation
	gradients, a = rnn_backward(X, Y, parameters, cache)

	# clip the gradients between -5 and 5 to avoid gradient exploding
	gradients = clip(gradients, 5)

	# update parameters
	parameters = update_parameters(parameters, gradients, learning_rate)

	return loss, gradients, a[len(X) - 1]
	

def model(data, ix_to_ch, ch_to_ix, number_interations=35000, n_a = 50, dino_names=7):
	'''
	Train the model and generate dinosaur names

	Arguments:
		data 		--
		ix_to_ch 	--
		ch_to_ix	--
		number_interations --
		n_a 		-- number of unit of RNN cell
		vocab_size 	--

	Returns:
		parameters 	-- learned parameters
	'''
	n_x, n_y = vocab_size, vocab_size

	parameters = initialize_parameters(n_a, n_x, n_y)

	examples = [x.lower().strip() for x in data]

	# Shuffle list of all dinosaur names
	np.random.seed(0)
	np.random.shuffle(examples)

	a_prev = np.zeros((n_a, 1))

	loss = get_initial_loss(vocab_size, dino_names)

	all_loss = [loss]

	for j in range(number_interations):
		idx = j % len(examples)

		# Prepare data
		single_example = examples[idx]
		single_example_ch = [c for c in single_example]
		single_example_ix = [ch_to_ix[i] for i in single_example_ch]
		X = [None] + single_example_ix
		Y = X[1:] + [ch_to_ix['\n']]		# because Y[0] needs to be equal to X[1]

		# perform optimization: Forwad --> Backward --> Update parameters
		curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

		# smooth loss
		loss = smooth(loss, curr_loss)

		# for debug
		all_loss.append(loss)

		# Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
		if j % 2000 == 0:
			print('---------------------')
			print('Iteration: %d, Loss: %f' % (j, loss))

			# The number of dinosaur names to print
			seed = 0
			for name in range(dino_names):
				# Sample indices and print them
				sampled_indices = sample(parameters, ch_to_ix, seed)
				print_sample(sampled_indices, ix_to_ch)

				seed += 1  # To get the same result (for grading purposes), increment the seed by one. 

			print('\n')

	plt.plot(np.array(all_loss))
	plt.show()


### HELPER funtion ###
def clip(gradients, maxValue):
	'''
	Clips the gradients' values between minimum and maximum.

	Arguments:
	gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
	maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

	Returns: 
	gradients -- a dictionary with the clipped gradients.
	'''

	dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

	### START CODE HERE ###
	# clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
	for gradient in [dWaa, dWax, dWya, db, dby]:
	    np.clip(gradient, -maxValue, maxValue, out=gradient)
	### END CODE HERE ###

	gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

	return gradients


### TESTING FUNCTIONS ###
def test_sample():
	'''
	Expected output:

	Sampling:
	list of sampled indices:
 		[12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, 7, 25, 6, 
 		25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 17, 24, 12, 13, 24, 0]
	list of sampled characters:
 		['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', 'u', 'n', 'c', 'b', 
 		'a', 'u', 'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', 'l', 'k', 'g', 'a', 'l', 
 		'j', 'b', 'g', 'g', 'k', 'q', 'x', 'l', 'm', 'x', '\n']
	'''
	np.random.seed(2)
	n_a = 100
	parameters = {}
	parameters['Waa'] = np.random.randn(n_a, n_a)
	parameters['Wax'] = np.random.randn(n_a, vocab_size)
	parameters['Wya'] = np.random.randn(vocab_size, n_a)
	parameters['ba'] = np.random.randn(n_a, 1)
	parameters['by'] = np.random.randn(vocab_size, 1)
	indices = sample(parameters, ch_to_ix, seed = 0)

	print(indices)
	print([ix_to_ch[i] for i in indices])


def test_optimize():
	'''
	Expected output:
		Loss = 126.503975722
		gradients["dWaa"][1][2] = 0.194709315347
		np.argmax(gradients["dWax"]) = 93
		gradients["dWya"][1][2] = -0.007773876032
		gradients["db"][4] = [-0.06809825]
		gradients["dby"][1] = [ 0.01538192]
		a_last[4] = [-1.]
	'''
	np.random.seed(1)
	vocab_size, n_a = 27, 100
	a_prev = np.random.randn(n_a, 1)
	Wax, Waa = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a)
	Wya = np.random.randn(vocab_size, n_a)
	b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
	parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
	X = [12,3,5,11,22,3]
	Y = [4,14,11,22,25, 26]

	loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
	print("Loss =", loss)
	print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
	print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
	print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
	print("gradients[\"db\"][4] =", gradients["db"][4])
	print("gradients[\"dby\"][1] =", gradients["dby"][1])
	print("a_last[4] =", a_last[4])


if __name__ == "__main__":
	# test_optimize()
	with open("dinos.txt") as f:
		data = f.readlines()
	model(data, ix_to_ch, ch_to_ix, number_interations=50000, n_a = 50)


