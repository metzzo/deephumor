from django.contrib import admin
from django.http import HttpResponseRedirect
from django.urls import path, reverse

from .models import Cartoon, FunninessAnnotation, ImageAnnotation, ImageAnnotationCollection, ImageAnnotationClass, \
    CartoonThemeClass

from django.utils.html import format_html


class ImageAnnotationAdmin(admin.StackedInline):
    class Media:
        js = (
            "/static/jquery.js",
            "/static/cropper.js",
            "/static/crop.js",
        )
        css = {
            'all': ("/static/cropper.css",)
        }
    model = ImageAnnotation
    extra = 0
    readonly_fields = ['cartoon_image',]
    fields = ['cartoon_image', 'dimensions', 'annotation_class',]

    def cartoon_image(self, obj):
        return format_html('<img src="{}" id="{}"'.format(obj.collection.cartoon.img.url, 'crop' + str(obj.pk)) +
                           '/>\n<script>\ninitCrop("crop' + str(obj.pk) + '");\n</script>')

    cartoon_image.short_description = 'Cartoon'

    def has_add_permission(self, request, obj=None):
        return False


class ImageAnnotationCollectionAdmin(admin.ModelAdmin):
    change_list_template = "image_annotation_changelist.html"
    change_form_template = "image_annotation_changeform.html"

    def original_cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.cartoon.original_img.url))
    original_cartoon_image.short_description = 'Cartoon'


    def original_cartoon_image_small(self, obj):
        return format_html('<img src="{}" style="width:100px; height:auto" />'.format(obj.cartoon.original_img.url))

    original_cartoon_image_small.short_description = 'Original Image'

    def punchline(self, obj):
        return format_html('<b>{}</b>'.format(obj.cartoon.punchline))
    punchline.short_description = 'Punchline'

    fields = ['original_cartoon_image', 'punchline', 'annotated_by', 'cartoon', 'annotated',]
    readonly_fields = ['annotated_by', 'original_cartoon_image', 'punchline', 'cartoon',]
    list_display = ['original_cartoon_image_small', 'punchline',]
    inlines = [ImageAnnotationAdmin]

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('annotate_next_image/', self.annotate_next_image),
        ]
        return my_urls + urls

    def add_annotation(self, obj, request):
        annotation = ImageAnnotation()
        annotation.collection = obj
        annotation.save()
        self.message_user(request, "New Annotation annotation")
        return HttpResponseRedirect(".")

    def annotate_next_image(self, request, obj=None):
        if obj is not None:
            obj.annotated = True
            obj.save()

        all_annotations = ImageAnnotationCollection.objects\
            .all()\
            .filter(annotated_by=request.user)
        annotation = all_annotations.filter(annotated=False).first()

        if annotation is None:
            # get cartoon which does not have annotation yet
            annotations = list(all_annotations)
            annotation_ids = list(map(lambda obj: obj.cartoon.id, annotations))
            unannotated_cartoons = Cartoon.objects.all().exclude(id__in=annotation_ids).exclude(relevant=False)
            selected_cartoon = unannotated_cartoons.first()
            if selected_cartoon is not None:
                print("make funniness annotation")
                annotation = ImageAnnotationCollection()
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
            return self.annotate_next_image(request, obj=obj)
        else:
            return super(ImageAnnotationCollectionAdmin, self).response_add(request, obj, post_url_continue)

    def response_change(self, request, obj, post_url_continue=None):
        if "_add-annotation" in request.POST:
            return self.add_annotation(request=request, obj=obj)
        elif '_addanother' in request.POST:
            return self.annotate_next_image(request, obj=obj)
        else:
            return super(ImageAnnotationCollectionAdmin, self).response_change(request, obj)

    def get_form(self, request, obj=None, **kwargs):
        form = super(ImageAnnotationCollectionAdmin, self).get_form(request, obj, **kwargs)
        return form

    def save_model(self, request, obj, form, change):
        obj.annotated_by = request.user
        super(ImageAnnotationCollectionAdmin, self).save_model(request, obj, form, change)



class CartoonAdmin(admin.ModelAdmin):
    change_form_template = "cartoon_changeform.html"
    class Media:
        js = (
            "/static/jquery.js",
            "/static/cropper.js",
            "/static/crop.js",
        )
        css = {
            'all': ("/static/cropper.css",)
        }
    change_list_template = "cartoon_changelist.html"

    def cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.img.url))

    cartoon_image.short_description = 'Cartoon'

    def original_cartoon_image(self, obj):
        return format_html('<img src="{}" id="{}"'.format(obj.original_img.url, 'crop' + str(obj.pk)) +
                           '/>\n<script>\ninitCrop("crop' + str(obj.pk) + '", true);\n</script>')

    original_cartoon_image.short_description = 'Original Image'

    def original_cartoon_image_small(self, obj):
        return format_html('<img src="{}" style="width:100px; height:auto" />'.format(obj.original_img.url))

    original_cartoon_image_small.short_description = 'Original Image'

    fields = ['cartoon_image', 'original_cartoon_image', 'custom_dimensions', 'punchline', 'relevant', 'annotated','is_multiple',]
    readonly_fields = ['cartoon_image', 'original_cartoon_image',]
    list_display = ['punchline', 'original_cartoon_image_small', 'relevant', 'annotated']

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

    def response_change(self, request, obj, post_url_continue=None):
        if '_annotate_next' in request.POST:
            obj.annotated = True
            obj.save()
            return self.process_next(request)
        elif '_addanother' in request.POST:
            return self.process_next(request)
        else:
            return super(CartoonAdmin, self).response_change(request, obj)

    def has_delete_permission(self, request, obj=None):
        return False

    def has_add_permission(self, request, obj=None):
        return False


class FunninessAnnotationAdmin(admin.ModelAdmin):
    change_list_template = "funniness_annotation_changelist.html"

    def original_cartoon_image(self, obj):
        return format_html('<img src="{}" />'.format(obj.cartoon.original_img.url))
    original_cartoon_image.short_description = 'Cartoon'


    def original_cartoon_image_small(self, obj):
        return format_html('<img src="{}" style="width:100px; height:auto" />'.format(obj.cartoon.original_img.url))

    original_cartoon_image_small.short_description = 'Original Image'

    def punchline(self, obj):
        return format_html('<b>{}</b>'.format(obj.cartoon.punchline))
    punchline.short_description = 'Punchline'

    fields = ['funniness', 'original_cartoon_image', 'punchline', 'annotated_by', 'cartoon', ]
    readonly_fields = ['annotated_by', 'original_cartoon_image', 'punchline',]
    list_display = ['punchline', 'original_cartoon_image_small', 'funniness',]

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
            return super(FunninessAnnotationAdmin, self).response_change(request, obj)

    def get_form(self, request, obj=None, **kwargs):
        form = super(FunninessAnnotationAdmin, self).get_form(request, obj, **kwargs)
        return form

    def save_model(self, request, obj, form, change):
        obj.annotated_by = request.user
        super(FunninessAnnotationAdmin, self).save_model(request, obj, form, change)

    def has_add_permission(self, request, obj=None):
        return True


class ImageAnnotationClassAdmin(admin.ModelAdmin):
    list_display = ['name',]

class CartoonThemeClassAdmin(admin.ModelAdmin):
    list_display = ['name',]


admin.site.register(Cartoon, CartoonAdmin)
admin.site.register(FunninessAnnotation, FunninessAnnotationAdmin)
admin.site.register(ImageAnnotationCollection, ImageAnnotationCollectionAdmin)
admin.site.register(ImageAnnotationClass, ImageAnnotationClassAdmin)
admin.site.register(CartoonThemeClass, CartoonThemeClassAdmin)
