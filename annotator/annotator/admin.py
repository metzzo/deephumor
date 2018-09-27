from django.contrib import admin
from django import forms
import random

from django.http import HttpResponseRedirect
from django.urls import path, reverse

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


def get_funniness_annotation_cartoon():
    cartoons = Cartoon.objects.all()
    return random.sample(list(cartoons), k=1)[0]


class CartoonImageForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(CartoonImageForm, self).__init__(*args, **kwargs)
        form.base_fields['cartoon'].initial = get_funniness_annotation_cartoon()
        self.fields['cartoon'] = forms.CharField()

    class Meta:
        model = FunninessAnnotation
        fields = ['funniness', 'cartoon',]


class FunninessAnnotationAdmin(admin.ModelAdmin):
    change_list_template = "annotation_changelist.html"

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
