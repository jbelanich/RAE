from numpy import *

import util

class RAETree:
	"""
	Autoencoder tree composed of RAETreeNodes.
	"""

	def __init__(self):
		print "test"


class RAETreeNode:
	"""
	Node in an autoencoder tree.  Each node has
	a reconstruction error, a classification, and
	a parent representation.

	Also has a reference to its two child RAETreeNodes.
	These may be None.
	"""

	def __init__(self, params, *args):
		"""
		Params of the form of the following tuple:
		(W1,b1,W2,b2,Wlabel)
		"""
		self.params = params

		if (len(args) > 1):
			(self.c1,self.c2) = args
			self.buildRepresentation()
		else:
			self.p = args[0]
			self.reconError = 0

	def buildRepresentation(self):
		#unpack parameters
		(W1,b1,W2,b2,Wlabel) = self.params

		#construct node
		self.p = tanh(W1.dot(concatenate((self.c1.p,self.c2.p))) + b1)
		reconstruction = W2.dot(self.p) + b2
		c1Prime = reconstruction[0:100]
		c2Prime = reconstruction[101:200]
		self.reconError = util.reconError(self.c1.p,self.c2.p,c1Prime,c2Prime)
		self.c = util.softmax(Wlabel.dot(self.p))

