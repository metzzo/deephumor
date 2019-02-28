from django.contrib.auth.models import User
from django.db import models
import os

from django.db.models import ImageField, TextField, BooleanField


def get_image_path(instance, filename):
    return os.path.join('photos', str(instance.id), filename)


def relevant_cartoon_queryset():
    return Cartoon.objects.all() \
                .exclude(relevant=False) \
                .exclude(is_multiple=True) \
                .exclude(punchline='')


class CartoonThemeClass(models.Model):
    name = models.CharField(default='', max_length=100)

    def __str__(self):
        return self.name


class Cartoon(models.Model):
    punchline = TextField(blank=True)
    img = ImageField(upload_to=get_image_path, blank=True, null=True)
    original_img = ImageField(upload_to=get_image_path, blank=True, null=True)
    name = TextField(default='')
    relevant = BooleanField(default=True)
    annotated = BooleanField(default=False)
    is_multiple = BooleanField(default=False)
    custom_dimensions = models.CharField(default='', max_length=100, blank=True)
    #consistent = BooleanField(default=True)
    duplicate_of = models.ForeignKey(
        'self',
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )

    @property
    def has_punchline(self):
        return len(self.punchline.strip()) > 0


class ImageAnnotationCollection(models.Model):
    annotated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        blank=True,
        null=True
    )
    cartoon = models.ForeignKey(
        Cartoon,
        on_delete=models.SET_NULL,
        blank=True,
        null=True
    )
    annotated = BooleanField(default=False)


class ImageAnnotationClass(models.Model):
    name = models.CharField(default='', max_length=100)

    def __str__(self):
        return self.name


class ImageAnnotation(models.Model):
    collection = models.ForeignKey(
        ImageAnnotationCollection,
        on_delete=models.SET_NULL,
        blank=True,
        null=True
    )
    annotation_class = models.ForeignKey(
        ImageAnnotationClass,
        on_delete=models.SET_NULL,
        blank=True,
        null=True
    )

    usable = models.BooleanField(
        default=True
    )

    dimensions = models.CharField(default='', max_length=100)


class FunninessAnnotation(models.Model):
    FUNNINESS_CHOICES = (
        (1, '1 ... not funny at all'),
        (2, '2 ... little smile'),
        (3, '3 ... big smile'),
        (4, '4 ... funny'),
        (5, '5 ... very funny'),
        (6, '6 ... hilarious'),
        (7, '7 ... very hilarious')

    )

    funniness = models.IntegerField(
        choices=FUNNINESS_CHOICES,
        blank=True,
        null=True
    )
    i_understand = models.BooleanField(
        default=True
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
