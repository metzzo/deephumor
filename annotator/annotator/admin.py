from django.contrib import admin
from django import forms
import random

from django.http import HttpResponseRedirect
from django.urls import path, reverse
from image_cropping import ImageCroppingMixin

from annotator import models
from .models import Cartoon, FunninessAnnotation, ImageAnnotation

from django.utils.html import format_html

class ImageAnnotationTest(ImageCroppingMixin, admin.ModelAdmin):
    pass
admin.site.register(ImageAnnotation, ImageAnnotationTest)


class ImageAnnotationAdmin(ImageCroppingMixin, admin.StackedInline):
    model = ImageAnnotation
    extra = 0


class CartoonAdmin(admin.ModelAdmin):
    change_list_template = "cartoon_changelist.html"

    def cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.img.url))

    cartoon_image.short_description = 'Cartoon'

    def original_cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.original_img.url))

    original_cartoon_image.short_description = 'Original Image'

    fields = ['cartoon_image', 'original_cartoon_image', 'punchline', 'relevant', 'annotated']
    readonly_fields = ['cartoon_image', 'original_cartoon_image',]
    inlines = [ImageAnnotationAdmin]

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('process_next/', self.process_next),
        ]
        return my_urls + urls

    def process_next(self, request):
        # check if there is some annotation without proper funniness
        cartoon = Cartoon.objects.all().filter().exclude(annotated=True).first()
        if cartoon is None:
            return HttpResponseRedirect("../")
        else:
            return HttpResponseRedirect(
                reverse('admin:%s_%s_change' % (cartoon._meta.app_label, cartoon._meta.model_name),
                        args=[cartoon.pk])
            )

    def has_delete_permission(self, request, obj=None):
        return False


admin.site.register(Cartoon, CartoonAdmin)


class FunninessAnnotationAdmin(admin.ModelAdmin):
    change_list_template = "funniness_annotation_changelist.html"

    def original_cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.cartoon.original_img.url))
    original_cartoon_image.short_description = 'Cartoon'

    def punchline(self, obj):
        return format_html('<b>{}</b>'.format(obj.cartoon.punchline))
    punchline.short_description = 'Punchline'

    fields = ['funniness', 'original_cartoon_image', 'punchline', 'annotated_by', 'cartoon', ]
    readonly_fields = ['annotated_by', 'original_cartoon_image', 'punchline',]

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('annotate_next_funniness/', self.annotate_next_funniness),
        ]
        return my_urls + urls

    def annotate_next_funniness(self, request):
        # check if there is some annotation without proper funniness
        all_annotations = FunninessAnnotation.objects.all().filter(annotated_by=request.user)
        annotation = all_annotations.filter(funniness=None).first()
        if annotation is None:
            # get cartoon which does not have annotation yet
            annotations = list(all_annotations)
            annotation_ids = list(map(lambda obj: obj.cartoon.id, annotations))
            unannotated_cartoons = Cartoon.objects.all().exclude(id__in=annotation_ids).exclude(relevant=False)
            selected_cartoon = unannotated_cartoons.first()
            if selected_cartoon is not None:
                print("make funniness annotation")
                annotation = FunninessAnnotation()
                annotation.cartoon = selected_cartoon
                annotation.annotated_by = request.user
                annotation.save()
            else:
                print("No cartoon left to annotate")
                return HttpResponseRedirect("../")

        return HttpResponseRedirect(
            reverse('admin:%s_%s_change' % (annotation._meta.app_label, annotation._meta.model_name),
                    args=[annotation.pk])
        )

    def response_add(self, request, obj, post_url_continue=None):
        if '_addanother' in request.POST:
            return self.annotate_next_funniness(request)
        else:
            return super(FunninessAnnotationAdmin, self).response_add(request, obj, post_url_continue)

    def response_change(self, request, obj, post_url_continue=None):
        if '_addanother' in request.POST:
            return self.annotate_next_funniness(request)
        else:
            return super(FunninessAnnotationAdmin, self).response_change(request, obj, post_url_continue)

    def get_form(self, request, obj=None, **kwargs):
        form = super(FunninessAnnotationAdmin, self).get_form(request, obj, **kwargs)
        return form

    def save_model(self, request, obj, form, change):
        obj.annotated_by = request.user
        super(FunninessAnnotationAdmin, self).save_model(request, obj, form, change)


admin.site.register(FunninessAnnotation, FunninessAnnotationAdmin)
