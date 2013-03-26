from numpy import *
from dictionary import dictionary
from genutil import *
from itertools import *

import mathutil as util

def genRandTree():
	params = genRandParams()
	sentence = 'testing on some random sentence'
	return RAETree(params, sentence)

class RAETree:
	"""
	Autoencoder tree composed of RAETreeNodes.
	"""

	def __unicode__(self):
		return self.root.__unicode__()

	def __iter__(self):
		"""
		Iterate over the tree's nodes depth first.
		"""
		return iter(self.root)

	def __init__(self, params, sentence):
		self.buildTree(params,sentence)
		self.backprop(params,sentence)

	def backprop(self, params, sentence):
		"""
		Compute the delta values for the gradient.
		"""
		self.root.backprop(params,sentence)
		return 0

	def buildTree(self, params, sentence):
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

			lowestErrorCandidate.c1.parent = lowestErrorCandidate
			lowestErrorCandidate.c2.parent = lowestErrorCandidate
			sentenceTree[lowestErrorIndex] = lowestErrorCandidate
			sentenceTree.pop(lowestErrorIndex+1)

		#done, we only have one element that is the root node of the tree
		self.root = sentenceTree[0]
		self.root.parent = None

	def reconError(self):
		"""
		Reconstruction error is the sum of the reconstruction errors
		of each individual node.
		"""
		sum = 0
		for node in self:
			sum += node.reconError
		return sum

	def numLeaves(self):
		return self.root.numLeaves()

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

	def backprop(self, params, sentence):
		"""
		Calculate delta values depending place in tree.
		"""

		#backprop doesn't need to calculate deltas for leaves
		if self.isLeaf():
			return

		(W1,b1,W2,b2,Wlabel) = self.params
		W2top = W2[0:100,:]
		W2bottom = W2[100:200,:]
		W1left = W1[:,0:100]
		W1right = W1[:,100:200]
		leftWeight = float(self.c1.numLeaves() + 1)/(self.c1.numLeaves() + self.c2.numLeaves() + 2)
		rightWeight = float(self.c2.numLeaves() + 1)/(self.c1.numLeaves() + self.c2.numLeaves() + 2)

		if self.isRoot():
			self.delta = -util.dtanh(self.a) * (leftWeight*W2top.transpose().dot(self.c1.p - self.c1Prime)+ rightWeight * W2bottom.transpose().dot(self.c2.p - self.c2Prime))
		else:
			if self.isLeftChild():
				properW = W1left
			else:
				properW = W1right
			self.delta = util.dtanh(self.a) * (properW.dot(self.parent.delta))

		self.c1.backprop(params,sentence)
		self.c2.backprop(params,sentence)

	def isLeftChild(self):
		return self.parent.c1 == self

	def buildRepresentation(self):
		#unpack parameters
		(W1,b1,W2,b2,Wlabel) = self.params

		#construct node
		self.c = concatenate((self.c1.p,self.c2.p))
		self.a = W1.dot(self.c) + b1
		self.p = tanh(self.a)
		#self.p = self.p / sum(self.p)
		reconstruction = W2.dot(self.p) + b2
		self.c1Prime = reconstruction[0:100]
		self.c2Prime = reconstruction[100:200]
		self.reconError = util.weightedReconError(
			self.c1.p,
			self.c2.p,
			self.c1Prime,
			self.c2Prime,
			self.c1.numLeaves()+1,
			self.c2.numLeaves()+1)
		self.classification = util.softmax(Wlabel.dot(self.p))

	def isRoot(self):
		return (self.parent == None)

	def isLeaf(self):
		return (self.c1 == None) or (self.c2 == None)

	def numLeaves(self):
		if self.isLeaf():
			return 0
		else:
			if self.c1.isLeaf() and self.c2.isLeaf():
				return 2
			elif self.c1.isLeaf() and not self.c2.isLeaf():
				return 1 + self.c2.numLeaves()
			elif not self.c1.isLeaf() and self.c2.isLeaf():
				return 1 + self.c1.numLeaves()
			else:
				return self.c1.numLeaves() + self.c2.numLeaves()


	def __unicode__(self):
		if self.isLeaf():
			return unicode(dictionary.rep2word(self.p))
		else:
			return self.c1.__unicode__() + ' ' + self.c2.__unicode__()
