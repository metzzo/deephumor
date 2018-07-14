from django.core.management.base import BaseCommand, CommandError
from annotator.tasks import import_comic

import os


class Command(BaseCommand):
    help = 'Imports the initial data set comics'

    def add_arguments(self, parser):
        parser.add_argument('path', nargs='+', type=str)

    def handle(self, *args, **options):
        input_directory = os.fsencode(options['path'][0])
        for file in os.listdir(input_directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                import_comic.delay(os.fsdecode(input_directory) + filename)
        self.stdout.write(self.style.SUCCESS('Tasks started'))
