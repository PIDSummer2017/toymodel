from toydatabasic import*
from toydatabasic import _choose_rectangle
from toydatabasic import _choose_triangle
from toydatabasic import _choose_vertical
from toydatabasic import _choose_horizontal
from toydatabasic import _image
from classification_image_gen import generate_noise
from classification_image_gen import randomize_labels_four
from toydata_varconfig import test_image
from numpy import random

#labeling still needs to be fixed!!!!!!!!!

def _choose_random_pixel(array):
    row = np.random.randint(img.SHAPE_SIZE, array.shape[0]-img.SHAPE_SIZE)
    col = np.random.randint(img.SHAPE_SIZE, array.shape[1]-img.SHAPE_SIZE)
    return row, col

def _randomize_shape(chance):
    """function to return True depending on 1/chance input. Takes an integer
    you really shouldn't have to mess with this."""
    z = np.random.randint(0, chance)
    if z < 1:
        return True
    else: return False

def _make_2d_labels(locs, nshapes = 4, maxmult = 5):
    """This function takes in an array with labels from all the *individual* shapes in an image,
    and generates a 2D array with shape type and multiplicities."""
    label_array = np.zeros([nshapes, maxmult])
    for _ in range(nshapes):
        tester = [0,0,0,0]
        tester[_] = 1
        mults = []
        colval = 0
        for element in locs:
            if element == tester:
                colval += 1
        label_array[_][colval] = 1
        mults.append(colval)

    return label_array

def _make_type_labels(locs, nshapes = img.NUM_SHAPES):

    label_array = [0,0,0,0]
    for _ in range(nshapes):
        tester = [0,0,0,0]
        tester[_] = 1
        for element in locs:
            if element == tester:
                label_array[_] = 1
    return label_array
    

def _randomize_type_labels():
    label_array = [0,0,0,0]
    z = random.randint(0, 5)
    for i in range(z):
        q = random.randint(0, 3)
        label_array[q] = 1
    return label_array


def _add_multiple_shapes_to(array, vals, nums = img.MULTIPLICITIES, probs = 3, types = img.ALLOWED, multlabels = img.MULTLABELS):
    """
    This function populates an array with random shapes.
    The inputs are as follows:
    1. an array of any dimension, to be populated with shapes
    2. a list to be populated with image labels
    3. a list of booleans corresponding to the shapes to be generated:
    [square, triangle, horizontal, vertical]
    4. a list of allowed maximum multiplicities for each shape:
    ex: [2, 2, 2, 2]
    5. an integer corresponding to the likelihood of generating a shape at a
    given location: for example, an input of 4 corresponds to a 1/4 chance
    that each allowed shape will actually exist"""
    locs = []
    if types[0]:
        for i in range(nums[0]):
            x, y = _choose_random_pixel(array)
            if _randomize_shape(probs):
                _choose_rectangle(x, y, array)
                if True:
                    locs.append([1,0,0,0])
    if types[1]:
        for i in range(nums[1]):
            x, y = _choose_random_pixel(array)
            if _randomize_shape(probs):
                _choose_triangle(x, y, array)
                if True:
                    locs.append([0,1,0,0])
    if types[2]:
        for i in range(nums[2]):
            x, y = _choose_random_pixel(array)
            if _randomize_shape(probs):
                 _choose_horizontal(x, y, array)
                 if True:
                    locs.append([0,0,1,0])
    if types[3]:
        for i in range(nums[3]):
            x, y = _choose_random_pixel(array)
            if _randomize_shape(probs):
                 _choose_vertical(x, y, array)
                 if True:  
                    locs.append([0,0,0,1])
    if not multlabels:
        vals.append(_make_type_labels(locs))

    else:
        vals.append(_make_2d_labels(locs))

class image_gen_counter:
    _counter_ = 0

#@ future me: make this more elegant, less inputs, etc. use classes? work on this tomorrow

def generate_training_images(num_images=1000,debug=0,bad_label = False, noise = 0, multlabels = img.MULTLABELS):
    bad_locations = []
    images = []
    vals = []

    for i in range(num_images):

        if debug:
            print 'Generating image',i

        mat = np.zeros([28,28]).astype(np.float32)

        _add_multiple_shapes_to(mat, vals)

        if noise:

            generate_noise(mat, noise)

        if debug>1:
            _image(mat)
            plt.savefig('image_%04d.png' % image_gen_counter._counter_)
            plt.close()

        mat = np.reshape(mat, (784))
        images.append(mat)

        image_gen_counter._counter_ +=1

    if bad_label:
        for loc in vals:
            if multlabels:
                bad_locations.append(randomize_labels_four())
            if not multlabels:
                bad_locations.append(_randomize_type_labels())
                          
 
    if bad_label:
        return images, bad_locations
    
    #print locations

    if multlabels:
        vals = np.reshape(vals, (4, num_images, 5))
    return images, vals

if __name__ == '__main__':
    batch = generate_training_images(bad_label = True, num_images = 200)
    print np.shape(batch[1])
    print batch[1]
