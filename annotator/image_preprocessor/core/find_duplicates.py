from pathlib import Path
from PIL import Image, ImageOps, ImageStat


def hash_image(image_path):
    img = Image.open(image_path).resize((8,8), Image.LANCZOS).convert(mode="L")
    mean = ImageStat.Stat(img).mean[0]
    return sum((1 if p > mean else 0) << i for i, p in enumerate(img.getdata()))


def find_duplicates(directory):
    pathlist = list(Path(directory).glob('**/original_cartoon.jpg'))
    hashes = {}
    count = len(pathlist)
    current = 0
    found_count = 0
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        hash = hash_image(path_in_str)
        if hash in hashes:
            found_count += 1
            hashes[hash] += [path_in_str]
        else:
            hashes[hash] = [path_in_str]
        current += 1
        print("Progress {0}/{1} Found: {2}".format(current, count, found_count))
    filtered = dict((key,value) for key, value in hashes.items() if len(value) > 1)
    return filtered

if __name__ == "__main__":
    print(find_duplicates('./media/'))
