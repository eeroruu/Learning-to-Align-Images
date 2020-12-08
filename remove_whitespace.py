# Remove whitespace from around the image

from PIL import Image, ImageChops
import glob


def trim(im):
    if im.mode == 'RGB':
        bg = Image.new(im.mode, im.size, (255, 255, 255))
    else:
        bg = Image.new(im.mode, im.size, 255)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 1.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


for f in glob.iglob("<Path to folder containing images>"):
    im = Image.open(f)
    im = trim(im)
    im.save("<Path to destination folder + name for image", "JPEG")