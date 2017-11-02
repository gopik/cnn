import string
from PIL import Image, ImageDraw, ImageFont
import utils
import numpy as np
import os.path
from keras.preprocessing.image import ImageDataGenerator
import tmp
import cv2
import random

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
    'dejavu/DejaVuSansMono.ttf',
    'ttf-ancient-scripts/Symbola605.ttf',
    'ttf-indic-fonts-core/MuktiNarrow.ttf',
    'liberation/LiberationSansNarrow-Regular.ttf',
    'msttcorefonts/Andale_Mono.ttf'
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
] + [ImageFont.truetype('ipython/SairaExtraCondensed-Regular.ttf', 32*4)]


def create_font_images(base_dir='fonts/base'):
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


def stretch_vertical(img, factor=1.8):
    img_resize = cv2.resize(img, (img.shape[1], int(img.shape[0] * factor)))
    padding = utils.get_padding(img_resize.shape[0], img_resize.shape[1], img.shape[0], img.shape[1])
    img_padded = np.pad(img_resize, padding, mode='constant')
    return cv2.resize(img_padded, (img.shape[1], img.shape[0]))


def stretch_horizontal(img, factor=1.5):
    img_resize = cv2.resize(img, (int(img.shape[1] * factor), img.shape[0]))
    padding = utils.get_padding(img_resize.shape[0], img_resize.shape[1], img.shape[0], img.shape[1])
    img_padded = np.pad(img_resize, padding, mode='constant')
    return cv2.resize(img_padded, (img.shape[1], img.shape[0]))


def gen_augmented_images():
    list_font_jpg = tmp.recursive_find_files('fonts/base', '.*jpeg')
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=False,
        fill_mode='nearest')

    for path in list_font_jpg:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cat = os.path.basename(os.path.dirname(path))
        file_basename = os.path.basename(path)
        x = img[np.newaxis, :, :, np.newaxis]
        i = 0
        target_dir = os.path.join('/home/gopik/github/cnn/fonts/train', cat)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for batch in datagen.flow(x, ['0'], batch_size=1, shuffle=True):
            aug_img = batch[0][0].reshape(40, 30)
            if i > 1000:
                print('Done generating %s' % cat)
                break  # otherwise the generator would loop indefinitely

            if random.uniform(0, 1) < 0.2:
                stretched_img = stretch_vertical(aug_img)
                file_path = os.path.join(target_dir, file_basename + '.vstretch.' + str(i) + '.jpeg')
                cv2.imwrite(file_path, stretched_img)
                i += 1

            if random.uniform(0, 1) < 0.2:
                stretched_img = stretch_horizontal(aug_img)
                file_path = os.path.join(target_dir, file_basename + '.hstretch.' + str[i] + '.jpeg')
                cv2.imwrite(file_path, stretched_img)
                i += 1

            file_path = os.path.join(target_dir, file_basename + str(i) + '.jpeg')
            cv2.imwrite(file_path, aug_img)
            i += 1


if __name__ == '__main__':
    print('Creating base font images')
    create_font_images()
    print('Creating augmented images for training')
    gen_augmented_images()