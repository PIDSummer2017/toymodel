import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import optimize
from toydata_varconfig import test_image

img = test_image()

print img

def _image(array):
    fig,ax = plt.subplots(figsize=(img.IMSIZE, img.IMSIZE),facecolor='w')
    plt.imshow(array,interpolation = 'nearest')
    # plt.show()

def _choose_triangle(x, y,array):
    if img.SHAPE_SIZE <= x <= len(array[0])-img.SHAPE_SIZE:
        if img.SHAPE_SIZE <= y <= len(array[1]) - img.SHAPE_SIZE:
            array[x:x+img.SHAPE_SIZE, y:y+img.SHAPE_SIZE] = 0
            array[x, y] += img.PIX_VAL
            q = 0
            while q <= img.SHAPE_SIZE:
                array[x-q:x+q, y-q] += img.PIX_VAL 
                q += 1

def _choose_rectangle(x, y, array):
    if x+img.SHAPE_SIZE <= len(array[0]):
        if y+img.SHAPE_SIZE <= len(array[1]):
            array[x:x+img.SHAPE_SIZE, y:y+img.SHAPE_SIZE] = 0
            points = [(x, y), (x+img.SHAPE_SIZE, y), (x, y+img.SHAPE_SIZE), (x+img.SHAPE_SIZE, y+img.SHAPE_SIZE)]
            start_pt, end_pt = min(points), max(points)
            array[start_pt[1]:end_pt[1]+1, start_pt[0]:end_pt[0]+1] += img.PIX_VAL

def _choose_horizontal(x, y, array):
    if y+img.SHAPE_SIZE <= len(array[1]):
        array[x:x+img.SHAPE_SIZE, y:y+img.SHAPE_SIZE] = 0
        array[x, y:y+img.SHAPE_SIZE] += img.PIX_VAL

def _choose_vertical(x, y, array):
    if x+img.SHAPE_SIZE <= len(array[0]):
        array[x:x+img.SHAPE_SIZE, y:y+img.SHAPE_SIZE] = 0
        array[x:x+img.SHAPE_SIZE, y] += img.PIX_VAL

if __name__ == '__main__':

    import random,sys
    dims=(15,15) # this is data (2d array) dimension (x,y)

    array=np.zeros(shape=dims).astype(np.float32)
    for _ in xrange(10000):
        array = array.reshape(dims)
        array[:] = 0.
        x = int(random.random() * (dims[0]-5))
        y = int(random.random() * dims[1])
        _choose_vertical(x,y,array)
        array = array.reshape(dims[0]*dims[1])
        if not (array>0).sum() == 5:
            print '# non-zero elements',(array>0).sum(),'(should be 5...)'
            print array>0
            sys.exit(1)
        if not (array[array>0] == 180.).astype(np.int32).sum() == 5:
            print 'Array contains element not exactly 180...'
            eys.exit(1)
    print 'Test successful'
    sys.exit(0)
