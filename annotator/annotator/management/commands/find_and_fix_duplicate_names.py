from django.core.management.base import BaseCommand, CommandError

from annotator.models import Cartoon
from image_preprocessor.core.find_duplicates import find_duplicates
import os

class Command(BaseCommand):
    help = 'Finds and fixes duplicate names'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        input_directory = "/Users/rfischer/CloudStation/MasterStudium/Diplomarbeit/Dataset/"

        duplicates = 0
        objects = list(Cartoon.objects.all())
        names = {}
        expected = set()
        obj_lookup = {}

        for file in os.listdir(input_directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                expected.add(os.fsdecode(input_directory) + filename)

        too_many = 0
        for obj in objects:
            if obj.name in names:
                if obj.relevant:
                    names[obj.name] += 1
                    duplicates += 1
                    obj.relevant = False
                    obj.duplicate_of = obj_lookup[obj.name]
                    obj.save()
            else:
                obj_lookup[obj.name] = obj
                names[obj.name] = 1

            if obj.name not in expected:
                too_many = 0

        missing = 0
        for e in expected:
            if e not in names:
                missing += 1



        print("Duplicates: {0}".format(duplicates))
        print("Missing: {0}".format(missing))
        print("Too many: {0}".format(too_many))

