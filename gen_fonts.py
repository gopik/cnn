import argparse
import string
from PIL import Image, ImageDraw, ImageFont
import utils
import numpy as np
import os.path
from keras.preprocessing.image import ImageDataGenerator
import tmp
import cv2
import random
from skimage import filters, morphology, io, transform

parser = argparse.ArgumentParser(description='Generate font data for training')
parser.add_argument('--refresh_base', help='generate base font images from font files', type=bool, default=False)
parser.add_argument('--train_dir', help='directory where training data is created', default='fonts/train')

args = parser.parse_args()

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


def create_font_images(base_dir='fonts/base'):
    for ch in chars_list:
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
    img_resize = transform.resize(img, (img.shape[0] * factor, img.shape[1]),
                                  preserve_range=True)
    padding = utils.get_padding(img_resize.shape[0], img_resize.shape[1], img.shape[0], img.shape[1])
    img_padded = np.pad(img_resize, padding, mode='constant')
    return np.uint8(transform.resize(img_padded, img.shape,
                                     preserve_range=True))


def stretch_horizontal(img, factor=1.8):
    img_resize = transform.resize(img, (img.shape[0], int(img.shape[1] *
                                                          factor)),
                                  preserve_range=True)
    padding = utils.get_padding(img_resize.shape[0], img_resize.shape[1], img.shape[0], img.shape[1])
    img_padded = np.pad(img_resize, padding, mode='constant')
    return np.uint8(transform.resize(img_padded, img.shape,
                                     preserve_range=True))

def crop_image(img):
	v = np.sum(img, axis=1)
	h = np.sum(img, axis=0)
	start_i = start_j = 0
	end_i = len(v)-1
	end_j = len(h)-1
	for i in range(len(v)):
		if v[i] > 50:
			start_i = i
			break

	for i in range(len(v)-1, 0, -1):
		if v[i] > 50:
			end_i = i
			break

	for i in range(len(h)):
		if h[i] > 50:
			start_j = i
			break

	for i in range(len(h)-1, 0, -1):
		if h[i] > 50:
			end_j = i
			break

	return np.uint8(transform.resize(img[start_i:end_i, start_j:end_j], img.shape, preserve_range=True))

def erode_image(img):
    """Erode and threshold an image."""

    return morphology.erosion(img)

def dilate_image(img):
    """Dilate and threshold an image."""

    return morphology.dilation(img)

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
        img = io.imread(path, as_grey=True) 
        cat = os.path.basename(os.path.dirname(path))
        file_basename, ext = os.path.basename(path).split('.')
        x = img[np.newaxis, :, :, np.newaxis]
        i = 0
        #print(args.train_dir)
        target_dir = os.path.join(args.train_dir, cat)
        #print(target_dir)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for batch in datagen.flow(x, ['0'], batch_size=1, shuffle=True):
            aug_img = np.uint8(batch[0][0].reshape(40, 30))
            if i > 200:
                print('Done generating %s' % cat)
                break  # otherwise the generator would loop indefinitely

            if random.uniform(0, 1) < 0.2:
                stretched_img = stretch_vertical(aug_img)
                file_path = os.path.join(target_dir, file_basename + '.vstretch.' + str(i) + '.jpeg')
                #io.imsave(file_path, stretched_img)
                cv2.imwrite(file_path, stretched_img)
                i += 1

            if random.uniform(0, 1) < 0.2:
                stretched_img = stretch_horizontal(aug_img)
                file_path = os.path.join(target_dir, file_basename + '.hstretch.' + str(i) + '.jpeg')
                #io.imsave(file_path, stretched_img)
                cv2.imwrite(file_path, stretched_img)
                i += 1

#            if random.uniform(0, 1) < 0.2:
#                dilated_image = dilate_image(aug_img)
#                file_path = os.path.join(target_dir, file_basename + '.dilate.' + str(i) + '.jpeg')
#                #io.imsave(file_path, dilated_image)
#                print(cv2.imwrite(file_path, dilated_image))
#                i += 1
#
#            if random.uniform(0, 1) < 0.1:
#                eroded_image = erode_image(aug_img)
#                file_path = os.path.join(target_dir, file_basename + '.erode.' + str(i) + '.jpeg')
#                #io.imsave(file_path, eroded_image)
#                print(cv2.imwrite(file_path, eroded_image))
#                i += 1

            if random.uniform(0, 1) < 0.2:
                cropped_image = crop_image(aug_img)
                file_path = os.path.join(target_dir, file_basename + '.crop.' +
                                         str(i) + '.jpeg')
                #io.imsave(file_path, cropped_image)
                cv2.imwrite(file_path, cropped_image)
                i += 1

            file_path = os.path.join(target_dir, file_basename + str(i) + '.jpeg')
            io.imsave(file_path, aug_img)
            i += 1


if __name__ == '__main__':
    if args.refresh_base:
        fonts_list = [
     #       ImageFont.truetype(file, 32*4) for file in map(lambda p: os.path.join(font_path_base, p), font_paths)
       ] + [ImageFont.truetype('fonts/SairaExtraCondensed-Regular.ttf', 32*4), ImageFont.truetype('fonts/BarlowCondensed-Regular.ttf', 32*4)]
        print('Creating base font images')
        create_font_images()

    print('Creating augmented images for training')
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    gen_augmented_images()
