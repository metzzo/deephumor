import os

from core.preprocess_image import preprocess_image


def convert_images(input_directory, output_directory):
    input_directory = os.fsencode(input_directory)
    results = []
    for file in os.listdir(input_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            out, punchline = preprocess_image(path=filename)
            results.append((out, punchline))


