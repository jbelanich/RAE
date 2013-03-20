import base64
from django.db import models

# Create your models here.
class Word(models.Model):
	word = models.CharField(max_length=255, unique=True)
	_vectorRep = models.TextField(db_column='vectorRep',blank=True)

	def setVectorRep(self, data):
		self._vectorRep = base64.encodestring(data)

	def getVectorRep(self):
		return base64.decodestring(self._vectorRep)

	vectorRep = property(getVectorRep,setVectorRep)