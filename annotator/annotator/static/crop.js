var loaded = false;
$(document).ready(function() {
  loaded = true;
});

function initCrop(image, isCartoon) {
    if (!loaded) {
        $(document).ready(function() {
            doCrop(image, isCartoon);
        });
    } else {
        doCrop(image, isCartoon);
    }
}

function doCrop(image, isCartoon) {
    var elem = $('#' + image);
    var field = null;
    if (isCartoon) {
        field = elem.parents('.field-original_cartoon_image').parent().find('.field-custom_dimensions div input');
    } else {
        field = elem.parents('.field-cartoon_image').parent().find('.field-dimensions div input');
    }
    var initial = [0, 0, 0, 0];
    try {
        initial = field.val().split(' ');
    } catch (ex) { }
    var cropper = new Cropper(elem.get(0), {
        viewMode: 2,
        rotatable: false,
        scalable: false,
        movable: false,
        zoomable: false,
        autoCrop: isCartoon ? field.val().length > 0 : true,
        data: {
            x: +initial[0],
            y: +initial[1],
            width: +initial[2],
            height: +initial[3],
        },
        crop(event) {
            var x = Math.max(0, ~~event.detail.x);
            var y = Math.max(0, ~~event.detail.y);
            var w = ~~event.detail.width;
            var h = ~~event.detail.height;

            field.val(x + ' ' + y + ' ' + w + ' ' + h);
        },
    });
}
