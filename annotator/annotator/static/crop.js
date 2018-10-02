var loaded = false;
$(document).ready(function() {
  loaded = true;
});

function initCrop(image) {
    if (!loaded) {
        $(document).ready(function() {
            doCrop(image);
        });
    } else {
        doCrop(image);
    }
}

function doCrop(image) {
    var elem = $('#' + image);
    var field = elem.parents('.field-cartoon_image').parent().find('.field-dimensions div input');
    var initial = field.val().split(' ');
    var cropper = new Cropper(elem.get(0), {
        viewMode: 2,
        rotatable: false,
        scalable: false,
        movable: false,
        zoomable: false,
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
