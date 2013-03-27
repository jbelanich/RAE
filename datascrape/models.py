from django.db import models

# Create your models here.
class Tweet(models.Model):
    created_at  = models.DateTimeField(auto_now_add=True)
    id_str            = models.CharField(unique=True, max_length=255)
    is_retweet        = models.BooleanField()
    text              = models.CharField(max_length=255)
    username          = models.CharField(max_length=255)
    screenname        = models.CharField(max_length=255)
    profile_image_url = models.URLField(max_length=255)

    def __unicode__(self):
        return u'@%s: %s' % (self.screenname, self.text)