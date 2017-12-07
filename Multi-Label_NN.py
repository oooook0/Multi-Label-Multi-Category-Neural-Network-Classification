import tensorflow as tf
import json
import numpy as np 

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

class multilabel_nn:
	
	def __init__(self, savefile, n_hidden_1 = 512,n_hidden_2 = 512, D = None, K = None):
		reset_graph()
		#reset the graph everytime reinitialize the class
		self.savefile= savefile
		self.n_hidden_1 = n_hidden_1
		self.n_hidden_2 = n_hidden_2
		if D and K:
			self.build(D, K)

	def build(self, D, K):

		self.inputs = tf.placeholder("float", [None, D])
		self.targets = tf.placeholder("float", [None, K])

		self.w = {
		'h1': tf.Variable(tf.random_normal([D, self.n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
		'out': tf.Variable(tf.random_normal([self.n_hidden_2, K]))
		}
		self.b = {
		'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
		'out': tf.Variable(tf.random_normal([K]))
		}

		self.saver = tf.train.Saver()

		def neural_net(x):
	        
			layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.w['h1']), self.b['b1']))
	        
			layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.w['h2']), self.b['b2']))
	    	
			out_layer = tf.matmul(layer_2, self.w['out']) + self.b['out']

			return out_layer

		logits = neural_net(self.inputs)
		self.predict_op = tf.nn.sigmoid(logits)

		loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.targets))
		
		return loss_op 

	def fit(self, X, Y, Xtest, Ytest, batch_size = 688, learning_rate = 0.01, result_freq = 2):
		N, D = X.shape
		N, K = Y.shape
		learning_rate = 0.01
		batch_size = 688
		n_batches = N // batch_size

		cost = self.build(D,K)

		train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		correct_pred = tf.equal(tf.argmax(self.predict_op, 1), tf.argmax(Ytest, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		init = tf.global_variables_initializer()

		with tf.Session() as session:
			session.run(init)
			for j in range(n_batches):
				Xbatch = X[j*batch_size:(j*batch_size + batch_size),]
				Ybatch = Y[j*batch_size:(j*batch_size + batch_size),]


				session.run(train_op, feed_dict={self.inputs: Xbatch, self.targets: Ybatch})
				if j % result_freq == 0:
					test_cost = session.run(cost, feed_dict={self.inputs: Xtest, self.targets: Ytest})
					Ptest = session.run(accuracy, feed_dict={self.inputs: Xtest})
					
					print("Cost / err at iteration j=%d: %.3f / %.3f" % (j, test_cost, Ptest))

			self.saver.save(session, self.savefile)

		self.D = D
		self.K = K

	def predict(self, X):
		
		with tf.Session() as session:
			self.saver = tf.train.import_meta_graph(self.savefile + '.meta')
			self.saver.restore(session, self.savefile)
			p = self.predict_op.eval(feed_dict={self.inputs: X})
		return p
"""
	def score(self, X, Y):
		return 1 - error_rate(self.predict(X), Y)
"""
	def save(self, filename):
		j = {
		  'D': self.D,
		  'K': self.K,
		  'model': self.savefile
		}
		with open(filename, 'w') as f:
			json.dump(j, f)
		print("model saved.")

	@staticmethod
	def load(filename):

		with open(filename) as f:
		  j = json.load(f)
		return multilabel_nn(j['model'], D=j['D'], K=j['K'])
		
