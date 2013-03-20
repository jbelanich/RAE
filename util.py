from numpy import *

def softmax(w,t=1.0):
	"""
	Computes the softmax of a nparray.  t is temperature
	of the softmax, defaulting to 1.0.
	"""
	unNorm = exp(array(w)/t)
	return unNorm/sum(unNorm)

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

def genRandParams():
	"""
	Returns random (W1,b1,W2,b2,Wlabel)
	"""
	return (
		random.rand(100,200),
		random.rand(100,1),
		random.rand(200,100),
		random.rand(200,1),
		random.rand(2,100)
		)

def genRandWord():
	"""
	Generates random words.
	"""
	return random.rand(100,1)
