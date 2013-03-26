from numpy import *

def softmax(w,t=1.0):
	"""
	Computes the softmax of a nparray.  t is temperature
	of the softmax, defaulting to 1.0.
	"""
	unNorm = exp(array(w)/t)
	return unNorm/sum(unNorm)

def dtanh(x):
	return 1 - tanh(x)**2

def reconError(c1,c2,c1Prime,c2Prime):
	"""
	Computes the reconstruction error.
	"""
	diff = concatenate((c1,c2)) - concatenate((c1Prime,c2Prime))
	return .5 * sqrt(
		sum(
			power(diff,2)
			)
		)

def weightedReconError(c1,c2,c1Prime,c2Prime,n1,n2):
	"""
	Computes the weighted reconstruction error.
	"""
	piece1 = (float(n1)/(n1+n2))*sum(power(c1-c1Prime,2))
	piece2 = (float(n2)/(n1+n2))*sum(power(c2-c2Prime,2))
	return piece1 + piece2

def isingle(thing):
	"""
	Iterator that iterates the single element provided.
	Useful for chaining.
	"""
	yield thing