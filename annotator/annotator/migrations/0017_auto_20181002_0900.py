# Generated by Django 2.1.2 on 2018-10-02 09:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0016_auto_20181002_0900'),
    ]

    operations = [
        migrations.RenameField(
            model_name='imageannotation',
            old_name='cartoon',
            new_name='collection',
        ),
    ]
