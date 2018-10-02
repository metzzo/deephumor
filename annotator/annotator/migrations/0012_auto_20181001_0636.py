# Generated by Django 2.0.7 on 2018-10-01 06:36

from django.db import migrations
import image_cropping.fields


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0011_auto_20181001_0635'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageannotation',
            name='min_free_cropping',
            field=image_cropping.fields.ImageRatioField('img', '100x100', adapt_rotation=False, allow_fullsize=False, free_crop=True, help_text=None, hide_image_field=False, size_warning=False, verbose_name='min free cropping'),
        ),
    ]