import toydatabasic
from toydatabasic import *
from toydatabasic import _choose_triangle
from toydatabasic import _choose_rectangle
from toydatabasic import _choose_horizontal
from toydatabasic import _choose_vertical
from toydatabasic import _image
import matplotlib as mpl
import numpy as np
mpl.use('agg')

def add_eight_shapes_to(array, locs, npoints = 1):
    """ this function adds one of eight different shape types to an array each time it is called:
    either one or two triangles, squares, horizontal lines or vertical lines. Images are labeled
    in a 8-dimensional list."""

    for i in range(npoints):
        row=int(random.uniform(5,array.shape[0]-5))
        col=int(random.uniform(5,array.shape[1]-5))
        array[row, col] = random.uniform(0,4)
        trianglocs = np.where(np.logical_and(0 < array, array < 1))
        rectlocs = np.where(np.logical_and(array >= 1, array <2))
        horlocs = np.where(np.logical_and(array >= 2, array < 3))
        vertlocs = np.where(array >=3)

        horxs = horlocs[0]
        horys = horlocs[1]

        vertxs = vertlocs[0]
        vertys = vertlocs[1]

        trixs = trianglocs[0]
        triys = trianglocs[1]

        rectxs = rectlocs[0]
        rectys = rectlocs[1]
        z = random.uniform(0,2)
        for i in range(len(rectxs)):
            x = rectxs[i]
            y = rectys[i]
            _choose_rectangle(x,y,array)
            if z < 1:
                _choose_rectangle(int(random.uniform(5, array.shape[0]-5)), int(random.uniform(5, array.shape[1]-5)), array)
                locs.append([0,1,0,0,0,0,0,0])
            else: locs.append([1,0,0,0,0,0,0,0])
        for i in range(len(triys)):
            x = trixs[i]
            y = triys[i]
            _choose_triangle(x,y,array)
            if z < 1:
                _choose_triangle(int(random.uniform(5, array.shape[0]-5)), int(random.uniform(5, array.shape[1]-5)), array)
                locs.append([0,0,0,1,0,0,0,0])
            else: locs.append([0,0,1,0,0,0,0,0])
        for i in range(len(horxs)):
            x = horxs[i]
            y = horys[i]
            _choose_horizontal(x,y,array)
            if z <1:
                _choose_horizontal(int(random.uniform(5, array.shape[0]-5)), int(random.uniform(5, array.shape[1]-5)), array)
                locs.append([0,0,0,0,0,1,0,0])
            else: locs.append([0,0,0,0,1,0,0,0])
        for i in range(len(vertys)):
            _choose_vertical(vertxs[i], vertys[i], array)
            if z <1:
                _choose_vertical(int(random.uniform(5, array.shape[0]-5)), int(random.uniform(5, array.shape[1]-5)), array)
                locs.append([0,0,0,0,0,0,0,1])
            else: locs.append([0,0,0,0,0,0,1,0])

def add_four_shapes_to(array, locs, npoints = 1):
    """ this function adds one of four shapes with a single multiplicity to
    an array each time it is called, and a 1 by 4 array labeling the shape called"""
    for i in range(npoints):
        row=int(random.uniform(5,array.shape[0]-5))
        col=int(random.uniform(5,array.shape[1]-5))
        array[row, col] = random.uniform(0,4)
        trianglocs = np.where(np.logical_and(0 < array, array < 1))
        rectlocs = np.where(np.logical_and(array >= 1, array <2))
        horlocs = np.where(np.logical_and(array >= 2, array < 3))
        vertlocs = np.where(array >=3)

        horxs = horlocs[0]
        horys = horlocs[1]

        vertxs = vertlocs[0]
        vertys = vertlocs[1]

        trixs = trianglocs[0]
        triys = trianglocs[1]

        rectxs = rectlocs[0]
        rectys = rectlocs[1]

        for i in range(len(rectxs)):
            x = rectxs[i]
            y = rectys[i]
            _choose_rectangle(x,y,array)
            locs.append([1,0,0,0])
        for i in range(len(triys)):
            x = trixs[i]
            y = triys[i]
            _choose_triangle(x,y,array)
            locs.append([0,1,0,0])
        for i in range(len(horxs)):
            x = horxs[i]
            y = horys[i]
            _choose_horizontal(x,y,array)
            locs.append([0,0,1,0])
        for i in range(len(vertys)):
            _choose_vertical(vertxs[i], vertys[i], array)
            locs.append([0,0,0,1])

def randomize_labels_eight():
    """
    This function returns an array of length 8 where only 1 element
    is set to 1 (randomly chosen) and the rest is set to 0.
    """
    labels = [0,0,0,0,0,0,0,0]
    z = random.randint(0, 7)
    labels[z] = 1
    return labels

def randomize_labels_four():
    """ this function returns an array of length 4, where one randomly chosen element is set to 1
    and the rest are zeros. This is useful to debug the single-multiplicity four shape image generator. """

    labels=np.random.choice([0,1], size=(4,))
    #labels = [0,0,0,0]
   # z = random.randint(0, 3)
   # labels[z] = 1
    return labels

def generate_noise(array, npoints):
    for x in xrange(npoints):
        row=int(random.uniform(0,array.shape[0]))
        col=int(random.uniform(0,array.shape[1]))
        array[row, col] = random.normal(128, 128)



class image_gen_counter:
    _counter_ = 0

def make_classification_images(num_images=10,debug=0,bad_label = False, multiplicities = True, noise = 0):
    """
    This function makes a set of variable classification images. The bad_label functionality randomizes the labels assigned
    to each image, to test training, while the debug function prints the images generated. The multiplicities corresponds
    to the numer of classification image types generated. If false, one of four basic shapes is generated and labeled.
    If multiplicities are included, eight basic images are generated. """
    locations = []
    bad_locations = []
    images = []

    for i in range(num_images):

        if debug:
            print 'Generating image',i

        mat = np.zeros([28,28]).astype(np.float32)

        if not multiplicities:
            add_four_shapes_to(mat, locations)

        if multiplicities:
            add_eight_shapes_to(mat, locations)

        generate_noise(mat, noise)

        if debug>1:
            _image(mat)
            plt.savefig('image_%04d.png' % image_gen_counter._counter_)
            plt.close()

        mat = np.reshape(mat, (784))
        images.append(mat)

        image_gen_counter._counter_ +=1

    if bad_label:
        for loc in locations:
            bad_locations.append(randomize_labels())

    if bad_label:
        return images, bad_locations
    #print locations
    return images, locations
