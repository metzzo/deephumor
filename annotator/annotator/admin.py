from django.contrib import admin
from django.http import HttpResponseRedirect
from django.urls import path

from .models import Cartoon, FunninessAnnotation

from django.utils.html import format_html


class CartoonAdmin(admin.ModelAdmin):
    def cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.img.url))

    cartoon_image.short_description = 'Cartoon'

    def original_cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.original_img.url))

    original_cartoon_image.short_description = 'Original Image'

    fields = ['cartoon_image', 'original_cartoon_image', 'punchline' ]
    readonly_fields = ['cartoon_image', 'original_cartoon_image' ]


admin.site.register(Cartoon, CartoonAdmin)


class FunninessAnnotationAdmin(admin.ModelAdmin):
    def original_cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.cartoon.original_img.url))

    original_cartoon_image.short_description = 'Cartoon'

    fields = ['original_cartoon_image', 'funniness']
    readonly_fields = ['original_cartoon_image', ]

    def save_model(self, request, obj, form, change):
        obj.annotated_by = request.user
        super(FunninessAnnotationAdmin, self).save_model(request, obj, form, change)


admin.site.register(FunninessAnnotation, FunninessAnnotationAdmin)
