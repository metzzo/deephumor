# Generated by Django 2.1.4 on 2019-02-27 13:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0003_funninessannotation_i_understand'),
    ]

    operations = [
        migrations.AddField(
            model_name='imageannotation',
            name='usable',
            field=models.BooleanField(default=True),
        ),
        migrations.AlterField(
            model_name='funninessannotation',
            name='funniness',
            field=models.IntegerField(blank=True, choices=[(1, '1 ... not funny at all'), (2, '2 ... little smile'), (3, '3 ... big smile'), (4, '4 ... funny'), (5, '5 ... very funny'), (6, '6 ... hilarious'), (7, '7 ... very hilarious')], null=True),
        ),
    ]
