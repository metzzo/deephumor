# Generated by Django 2.1.2 on 2018-10-01 19:50

import annotator.models
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0013_auto_20181001_0638'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imageannotation',
            name='min_free_cropping',
        ),
    ]