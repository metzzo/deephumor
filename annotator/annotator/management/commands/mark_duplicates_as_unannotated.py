from django.core.management.base import BaseCommand, CommandError

from annotator.models import Cartoon
from annotator.tasks import import_comic

import os


class Command(BaseCommand):
    help = 'Marks duplicates as not annotated.'

    def handle(self, *args, **options):
        cartoons = list(Cartoon.objects.all())
        for c in cartoons:
            if c.duplicate_of is not None:
                c.annotated = False
                c.save()