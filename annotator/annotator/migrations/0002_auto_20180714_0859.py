# Generated by Django 2.0.7 on 2018-07-14 08:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='cartoon',
            old_name='profile_image',
            new_name='img',
        ),
    ]
