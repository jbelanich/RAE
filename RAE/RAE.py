from numpy import *
from dictionary import dictionary
from itertools import *

import util

class RAETree:
	"""
	Autoencoder tree composed of RAETreeNodes.
	"""

	def __unicode__(self):
		return self.root.__unicode__()

	def __iter__(self):
		return iter(self.root)

	def __init__(self, params, sentence):
		"""
		Build the tree!
		"""
		S = dictionary.sentence2reps(sentence)
		sentenceTree = []
		for i in range(0,len(S)):
			sentenceTree.append(RAETreeNode(params,S[i]))

		while (len(sentenceTree) > 1):
			lowestError = float("inf")
			lowestErrorCandidate = None
			lowestErrorIndex = 0
			for i in range(0,len(sentenceTree)-1):
				candidate = RAETreeNode(params,sentenceTree[i],sentenceTree[i+1])
				if (lowestError > candidate.reconError):
					lowestError = candidate.reconError
					lowestErrorCandidate = candidate
					lowestErrorIndex = i

			sentenceTree[lowestErrorIndex] = lowestErrorCandidate
			sentenceTree.pop(lowestErrorIndex+1)

		#done, we only have one element that is the root node of the tree
		self.root = sentenceTree[0]

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
			self.c1 = None
			self.c2 = None
			self.reconError = 0

	def __iter__(self):
		if not self.isLeaf():
			return chain(util.isingle(self), iter(self.c1), iter(self.c2))
		else:
			return util.isingle(self)

	def buildRepresentation(self):
		#unpack parameters
		(W1,b1,W2,b2,Wlabel) = self.params

		#construct node
		self.p = tanh(W1.dot(concatenate((self.c1.p,self.c2.p))) + b1)
		self.p = self.p / sum(self.p)
		reconstruction = W2.dot(self.p) + b2
		c1Prime = reconstruction[0:100]
		c2Prime = reconstruction[100:200]
		self.reconError = util.reconError(self.c1.p,self.c2.p,c1Prime,c2Prime)
		self.c = util.softmax(Wlabel.dot(self.p))


	def isLeaf(self):
		return (self.c1 == None) or (self.c2 == None)

	def __unicode__(self):
		if self.isLeaf():
			return unicode(dictionary.rep2word(self.p))
		else:
			return self.c1.__unicode__() + ' ' + self.c2.__unicode__()


