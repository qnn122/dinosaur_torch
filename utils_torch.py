import numpy as np
import torch

def sample(model, ch_to_ix, vocab_size, seed):
	'''
	Sampling a sequence of characters based on the probabilities outputed by a RNN

	Arguments:
		paramenters	-- dictionary containing weights of the RNN: Waa, Wax, Way
		ch_to_ix 	-- dictionary mapping index to characters
		seed 		-- for testing purpose
		
	Returns
		indices 	-- list of length n containing the indices of generated characters
	'''
	indices = []
	newline_character = ch_to_ix['\n']
	
	# initialise x and a
	x = torch.zeros(1, vocab_size)
	a_prev = model.init_a0()

	idx = -1
	counter = 0 

	# generate new word, loop until meet new line character or reach 50 chars
	while (idx != newline_character and counter != 50):
		# calc y
		with torch.no_grad():
			a, y = model(x, a_prev)

		# for testing purpose
		np.random.seed(counter + seed)

		# select idx from y (probabilities)
		y = y.detach().numpy().ravel()
		idx = np.random.choice(range(len(y)), p=y)

		# one-hot encoding x
		x = torch.zeros(1, vocab_size)
		x[0][idx] = 1

		a_prev = a

		indices.append(idx)

		# increase counter
		counter += 1

	if counter == 50:
		indices.append(ch_to_ix['\n'])

	return indices

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print ('%s' % (txt, ), end='')

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def get_initial_loss(vocab_size, seq_length):
    return -torch.log(torch.tensor(1.0/vocab_size))*seq_length

### Test functions ###
def test_sample():
	data  = open('dinos.txt', 'r').read()
	data = data.lower()
	chars = list(set(data))
	data_size, vocab_size = len(data), len(chars)
	chars = sorted(chars)
	ch_to_ix = {ch:i for i, ch in enumerate(chars)} 
	ix_to_ch = {i:ch for i, ch in enumerate(chars)}

	from dinosaur_training_torch import RNN
	hidden_size = 100 	# = n_a
	data_size = 27		# n_x
	output_size = 27 	# n_x = n_y = 27
	rnn_dino = RNN(data_size, hidden_size, output_size)
	sampled_indices = sample(rnn_dino, ch_to_ix, data_size, 0)
	print_sample(sampled_indices, ix_to_ch)