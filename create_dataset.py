# Create training, validation or testing dataset for the network

from PIL import Image
from random import randrange
import numpy as np
import glob

# First number is the total number of images in the dataset
images = np.full((590, 128, 128, 2), 0)
transition_values = []

# i is for how many times to loop through the original images which are used to generate pairs
# j is for managing which images of the folder end up in the dataset. This is done so that testing and validation
# datasets will have pairs generated from different images

for i in range(0, 10):
    j = 0
    for f in glob.iglob("<Path to image folder>"):
        if j < 59:
            im = Image.open(f)
            im = im.convert("L")
            w, h = im.size

            # Randomize y-transtion, in this case between -10 and 10
            y_transition = randrange(1, 22) - 11
            y_start_point = 0

            # Check that there is enough
            while y_start_point == 0:
                if h-128-1-10 > 10:
                    y_start_point = randrange(0+10, h-128-1-10)
                else:
                    new_size = (int(1.5 * w), int(1.5 * h))
                    im = im.resize(new_size, Image.ANTIALIAS)
                    w, h = im.size

            up1 = y_start_point
            up2 = y_start_point + y_transition
            down1 = y_start_point + 128
            down2 = y_start_point + 128 + y_transition


            # Randomize x-transtion
            x_transition = randrange(1, 22) - 11
            if w-128-1-10 > 10:
                x_start_point = randrange(0+10, w-128-1-10)
            else:
                new_size = (int(1.5 * w), int(1.5 * h))
                im = im.resize(new_size, Image.ANTIALIAS)
                w, h = im.size
                x_start_point = randrange(0+10, w-128-1-10)

            left1 = x_start_point
            left2 = x_start_point + x_transition
            right1 = x_start_point + 128
            right2 = x_start_point + 128 + x_transition

            # Transition values contain information on how im2 should be moved according to image coordinates to align
            # it with the original

            '''
            For the resized method:
            im = im.resize((128+2*20, 128+2*20))
            '''

            temp = im.copy()
            im1 = temp.crop((left1, up1, right1, down1))
            im2 = im.crop((left2, up2, right2, down2))

            im1_array = np.array(im1)
            im2_array = np.array(im2)
            images[j+59*i, :, :, 0] = im1_array[:, :]
            images[j+59*i, :, :, 1] = im2_array[:, :]

            transition_values.append([y_transition, x_transition])
        j += 1

np.savez('<Path + name for the dataset>', images=images, values=transition_values)







