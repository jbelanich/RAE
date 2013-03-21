from numpy import *

def genRandParams():
	"""
	Returns random (W1,b1,W2,b2,Wlabel)
	"""
	return (
		random.randn(100,200),
		random.randn(100,1),
		random.randn(200,100),
		random.randn(200,1),
		random.randn(2,100)
		)

def genRandWord():
	"""
	Generates random words.
	"""
	return random.randn(100,1)

def genRandSentence(length):
	"""
	Generates length number of words in a list.

	Deprecated!  Do not use, reverse lookup
	for these words will result in Nones coming
	from the dictionary.
	"""
	sentence = []
	for i in range(0,length):
		sentence.append(genRandWord())
	return sentence

