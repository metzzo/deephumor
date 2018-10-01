from django.core.management.base import BaseCommand, CommandError

from annotator.models import Cartoon
from annotator.tasks import import_comic

import os


class Command(BaseCommand):
    help = 'Resets the annotated comics'

    def handle(self, *args, **options):
        Cartoon.objects.all().update(annotated=False)