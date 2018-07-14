from django.db import models
import os

from django.db.models import ImageField, TextField


def get_image_path(instance, filename):
    return os.path.join('photos', str(instance.id), filename)


class Cartoon(models.Model):
    punchline = TextField()
    img = ImageField(upload_to=get_image_path, blank=True, null=True)
    original_img = ImageField(upload_to=get_image_path, blank=True, null=True)
    name = TextField(default='')
