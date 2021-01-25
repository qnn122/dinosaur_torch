'''
TRAINING DINOSAUR NAME GENERATION USING PYTORCH

References:
	- https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#Creating-the-Network (structure)
	- https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html (train_func(), )
	- https://pytorch.org/tutorials/beginner/transformer_tutorial.html (init_weights(), ) 
	- https://pytorch.org/docs/stable/nn.init.html
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import sample, print_sample
from torch.autograd import Variable

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
data  = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
chars = sorted(chars)
ch_to_ix = {ch:i for i, ch in enumerate(chars)} 
ix_to_ch = {i:ch for i, ch in enumerate(chars)}

with open("dinos.txt") as f:
		data = f.readlines()
examples = [x.lower().strip() for x in data]


### MODEL ###
class RNN(nn.Module):
	def __init__(self, data_size, hidden_size, output_size):
		'''
		Arguments:
			data_size 	-- dimension of X, n_x
			hidden_size	-- dimention of hidden a, n_a
			output_size -- vocab_size 
		'''
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.i2h = nn.Linear(data_size + hidden_size, hidden_size) 	# Wax
		self.h2o = nn.Linear(hidden_size, output_size) 				# Wya

	def forward(self, data, last_hidden):
		'''
		Equivalent to rnn_step_forward()
		'''
		input = torch.cat((data, last_hidden), 1)
		hidden = torch.tanh(self.i2h(input))
		output = torch.softmax(self.h2o(hidden), dim=1)
		return hidden, output

	def init_weights(self):
		coeff = 0.01
		self.i2h.weight = nn.init.normal_(self.i2h.weight) * coeff
		nn.init.zeros_(self.i2h.bias) # or self.i2o.bias.data.zero_()
		self.h2o.weight = nn.init.normal_(self.h2o.weight) * coeff
		nn.init.zeros_(self.h2o.bias)

	def init_a0(self):
		'''
		create a0

		Returns:
			a0 	-- size (n_a)
		'''
		return torch.zeros(1, self.hidden_size)

# Initialize model
hidden_size = 100 	# = n_a
data_size = 27		# n_x
output_size = 27 	# n_x = n_y = 27
lr = 0.01
rnn_dino = RNN(data_size, hidden_size, output_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer  = torch.optim.SGD(rnn_dino.parameters(), lr=lr)

### TRAINER FUNCTIONS ###
def optimize(X, Y, a_prev):
	'''
	Returns:
		output
		loss
	'''
	loss = 0
	optimizer.zero_grad()

	x = {}

	# rnn_forward
	for t in range(len(X)):
		# hot encode X and y
		x[t] = np.zeros((vocab_size, 1))
		if (X[t] != None):
			x[t][X[t]] = 1

		# tensorise
		x_ts = torch.from_numpy(x[t]).to(torch.float32)
		y_ts = torch.tensor([Y[t]]) # IMPORTANT!

		# forwarding
		a, output = rnn_dino(torch.transpose(x_ts, 0, 1), a_prev)	

		# backward and update
		l = criterion(output, y_ts)
		loss += l
		a_prev = Variable(a, requires_grad=True)
		# a = a_prev # TODO: resolve this. Without it: Trying to backward through the graph a second time, but the buffers have already been freed.

	# backprobagation 
	loss.backward()

	# Clip gradients. IMPORTANT step
	'''
	clipping_value = 5
	torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
	'''

	# update parameters
	optimizer.step()

	return loss.item() / len(X), a

def train(data, number_iterations=35000, dino_names=7):
	'''
	Train the model and generate dinosaur names

	Arguments:
		data 		--
		ix_to_ch 	--
		ch_to_ix	--
		number_iterations --
		n_a 		-- number of unit of RNN cell
		vocab_size 	--

	Returns:
		parameters 	-- learned parameters
	'''
	# rnn_dino.init_weights()
	a_prev = rnn_dino.init_a0() 	# initialize parameters

	all_loss = []

	examples = [x.lower().strip() for x in data]

	# mini batch
	for j in range(number_iterations):
		# print(j)
		idx = j % len(examples)
		X, Y = random_pair(examples, idx)

		# Optimize parameter
		loss, a = optimize(X, Y, a_prev)
		a_prev = Variable(a, requires_grad=True)
		
		# for debug
		all_loss.append(loss)

		# Print
		# print('iter: {} | loss: {}'.format(j, loss))

		# Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
		if j % 2000 == 0:
			print('---------------------')
			print('Iteration: %d, Loss: %f' % (j, loss))

			# The number of dinosaur names to print
			seed = 0
			for name in range(dino_names):
				# Sample indices and print them
				sampled_indices = sample(rnn_dino, ch_to_ix, data_size, seed)
				print_sample(sampled_indices, ix_to_ch)
				seed += 1  # To get the same result (for grading purposes), increment the seed by one. 

			print('\n')

	plt.plot(np.array(all_loss))
	plt.show()

#### HELPERS ####
def random_pair(examples, idx):
	# Prepare data
	single_example = examples[idx]
	single_example_ch = [c for c in single_example]
	single_example_ix = [ch_to_ix[i] for i in single_example_ch]
	X = [None] + single_example_ix
	Y = X[1:] + [ch_to_ix['\n']]		# because Y[0] needs to be equal to X[1]
	return X, Y

if __name__ == '__main__':
	# test_optimize()
	with open("dinos.txt") as f:
		data = f.readlines()
	train(data, number_iterations=10000)

