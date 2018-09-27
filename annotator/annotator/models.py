from django.contrib.auth.models import User
from django.db import models
import os

from django.db.models import ImageField, TextField, BooleanField


def get_image_path(instance, filename):
    return os.path.join('photos', str(instance.id), filename)


class Cartoon(models.Model):
    punchline = TextField()
    img = ImageField(upload_to=get_image_path, blank=True, null=True)
    original_img = ImageField(upload_to=get_image_path, blank=True, null=True)
    name = TextField(default='')
    relevant = BooleanField(default=True)


class FunninessAnnotation(models.Model):
    FUNNINESS_CHOICES = (
        (1, '1 ... not funny at all'),
        (2, '2 ... moderately funny'),
        (3, '3 ... less than funny'),
        (4, '4 ... funny'),
        (5, '5 ... more than funny'),
        (6, '6 ... hilarious'),
        (7, '7 ... more than hilarious')

    )

    funniness = models.IntegerField(
        choices=FUNNINESS_CHOICES,
        blank=True,
        null=True
    )

    annotated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        blank=True,
        null=True
    )

    cartoon = models.ForeignKey(
        Cartoon,
        on_delete=models.CASCADE,
    )
