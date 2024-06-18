from django.db import models


# Create your models here.

class Users(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_fio = models.CharField(max_length=100)
    user_photo = models.ImageField()

