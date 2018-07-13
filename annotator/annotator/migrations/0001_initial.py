# Generated by Django 2.0.7 on 2018-07-13 17:52

import annotator.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Cartoon',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('punchline', models.TextField()),
                ('profile_image', models.ImageField(blank=True, null=True, upload_to=annotator.models.get_image_path)),
            ],
        ),
    ]
