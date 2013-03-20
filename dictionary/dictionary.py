import os

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

def translate(queryWord):
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

	return fromstring(wordResult.vectorRep)



