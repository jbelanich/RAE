from scipy.optimize import *
from numpy import *


def unpackParams(arrayRepParams):
	W1 = zeros((100,200))
	b1 = zeros((100,1))
	b2 = zeros((200,1))
	W2 = zeros((200,100))
	Wlabel = zeros((2,100))

	offset = 0
	#construct W1
	for i in range(0,100):
		for j in range(0,200):
			W1[i,j] = arrayRepParams[offset + i*200 + j]

	offset += 20000

	#construct b1
	for i in range(0,100):
		b1[i,0] = arrayRepParams[offset + i]

	offset += 100

	#construct W2
	for i in range(0,200):
		for j in range(0,100):
			W2[i,j] = arrayRepParams[offset + i*100 + j]

	offset += 20000

	#construct b2
	for i in range(0,200):
		b2[i,0] = arrayRepParams[offset+i]

	offset += 200

	#construct Wlabel
	for i in range(0,2):
		for j in range(0,100):
			Wlabel[i,j] = arrayRepParams[offset + i*100 + j]

	return (W1,b1,W2,b2,Wlabel)

def packParams(params):
	(W1,b1,W2,b2,Wlabel) = params

	arrayRepParams = zeros(40500)

	offset = 0
	#construct W1
	for i in range(0,100):
		for j in range(0,200):
			arrayRepParams[offset + i*200 + j] = W1[i,j]

	offset += 20000

	#construct b1
	for i in range(0,100):
		arrayRepParams[offset + i] = b1[i,0]

	offset += 100

	#construct W2
	for i in range(0,200):
		for j in range(0,100):
			arrayRepParams[offset + i*100 + j] = W2[i,j]

	offset += 20000

	#construct b2
	for i in range(0,200):
		arrayRepParams[offset+i] = b2[i,0]

	offset += 200

	#construct Wlabel
	for i in range(0,2):
		for j in range(0,100):
			arrayRepParams[offset + i*100 + j] = Wlabel[i,j]

	return arrayRepParams

def unregObjectiveFunction(params,unlabeledData):
	"""
	unlabeledData is simply a list of sentences.

	Very naive object just for testing
	"""
	errorSum = 0
	for sentence in unlabeledData:
		tree = RAETree(params,sentence)
		errorSum += tree.reconError()

	return errorSum/len(unlabeledData)