import random,sys
from toydatabasic import *
from toydata_varconfig import test_image

img = test_image()

dims=(img.IMSIZE,img.IMSIZE) # this is data (2d array) dimension (x,y)

array=np.zeros(shape=dims).astype(np.float32)
for _ in xrange(10000):
    array = array.reshape(dims)
    array[:] = 0.
    x = int(random.random() * (dims[0]-img.SHAPE_SIZE))
    y = int(random.random() * dims[1]-img.SHAPE_SIZE)
    _choose_vertical(x,y,array)
    array = array.reshape(dims[0]*dims[1])
    if not (array>0).sum() == img.SHAPE_SIZE:
        print '# non-zero elements',(array>0).sum(),'(should be 5...)'
        print array>0
        sys.exit(1)
    if not (array[array>0] == img.PIX_VAL).astype(np.int32).sum() == img.SHAPE_SIZE:
        print 'Array contains element not exactly 180...
        eys.exit(1)
print 'Test successful'
sys.exit(0)
