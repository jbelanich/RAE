import os
import base64
import string

from models import *
from RAE.util import *

"""

Words are stored in a vocabulary V and their representations
are stored in a matrix L.

During training, we want to be able to add/remove words
from the dictionary, and have the dictionary translate
sentences to their vector representations.

During testing and deployment, we want the dictionary
to be able to add words to the dictionary that it
has never seen before.

The dictionary is maintained as a sqlite database.

"""

def reps2sentence(reps):
	return string.join(reps2words(reps))

def reps2words(reps):
	words = []
	for rep in reps:
		words.append(rep2word(rep))

	return words

def sentence2reps(sentence):
	return words2reps(sentence.split())

def words2reps(words):
	reps = []
	for word in words:
		reps.append(word2rep(word))

	return reps

def rep2word(queryVector):
	try:
		wordResult = Word.objects.get(_vectorRep=base64.encodestring(
			queryVector.tostring()))
	except Word.DoesNotExist:
		return None

	return wordResult.word

def word2rep(queryWord):
	"""
	Return the vector representation of a word.  If the
	word doesn't exist in the dictionary, a new representation
	is created and inserted in the dictionary.
	"""
	try:
		wordResult = Word.objects.get(word=queryWord)
	except Word.DoesNotExist:
		wordResult = Word(word=queryWord,vectorRep=genRandWord().tostring())
		wordResult.save()

	return reshape(fromstring(wordResult.vectorRep), (100,1))



