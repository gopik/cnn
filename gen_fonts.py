import string
from PIL import Image, ImageDraw, ImageFont
import utils
import numpy as np
import os.path

# TODO: Add flags for file paths and image sizes.

chars = string.ascii_uppercase + string.digits

font_path_base = '/usr/share/fonts/truetype/'
font_paths = [
    'msttcorefonts/Trebuchet_MS.ttf',
    'msttcorefonts/Courier_New.ttf',
    'msttcorefonts/Times_New_Roman.ttf',
    'msttcorefonts/Verdana.ttf',
    'msttcorefonts/Arial.ttf',
    'msttcorfonts/Georgia.ttf',
    'freefont/FreeMono.ttf',
    'freefont/FreeSans.ttf',
    'freefont/FreeSerif.ttf',
    'dejavu/DejaVuSans.ttf',
    'dejavu/DejaVuSerif.ttf',
    'dejavu/DejaVuSansMono.ttf'
]


def max_height(font):
    """Maximum height of a character in the input font"""
    return max(font.getsize(char)[1] for char in chars)


def get_font_img(font, ch, height):
    """Create an image of the input character. Image height is provided as argument."""
    width, _ = font.getsize(ch)
    img = Image.new('RGBA', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (255, 255, 255), font=font)
    return img


def get_file_name(fnt, ch):
    """Filename for the character (using font name and character name)"""
    name, _ = fnt.getname()
    name = name.replace(' ', '_')
    return name + '_' + ch + '.jpeg'


fonts_list = [
    ImageFont.truetype(file, 32*4) for file in map(lambda p: os.path.join(font_path_base, p), font_paths)
]


# Base directory where font images need to be created.
base_dir = '/home/gopik/github/cnn/fonts/base'
for ch in chars:
    target_dir = os.path.join(base_dir, ch)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for i in range(len(fonts_list)):
        fnt = fonts_list[i]
        target_file = os.path.join(target_dir, get_file_name(fnt, ch))
        h = max_height(fnt)
        img = get_font_img(fnt, ch, h)
        padding = utils.get_padding(img.height, img.width, 40, 30)
        gray_scale = np.array(img.convert('L'))
        img_pad = np.pad(gray_scale, padding, mode='constant', constant_values=0)
        img_padded_pil = Image.fromarray(img_pad).resize((30, 40), Image.ANTIALIAS)
        img_padded_pil.save(target_file)


