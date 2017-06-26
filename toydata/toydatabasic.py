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
            return True
    else: return False

def _choose_rectangle(x, y, array):
    if x+img.SHAPE_SIZE <= len(array[0]):
        if y+img.SHAPE_SIZE <= len(array[1]):
            array[x:x+img.SHAPE_SIZE, y:y+img.SHAPE_SIZE] = 0
            points = [(x, y), (x+img.SHAPE_SIZE, y), (x, y+img.SHAPE_SIZE), (x+img.SHAPE_SIZE, y+img.SHAPE_SIZE)]
            start_pt, end_pt = min(points), max(points)
            array[start_pt[1]:end_pt[1]+1, start_pt[0]:end_pt[0]+1] += img.PIX_VAL
            return True
    else: return False

def _choose_horizontal(x, y, array):
    if y+img.SHAPE_SIZE <= len(array[1]):
        array[x:x+img.SHAPE_SIZE, y:y+img.SHAPE_SIZE] = 0
        array[x, y:y+img.SHAPE_SIZE] += img.PIX_VAL
        return True
    else: return False

def _choose_vertical(x, y, array):
    if x+img.SHAPE_SIZE <= len(array[0]):
        array[x:x+img.SHAPE_SIZE, y:y+img.SHAPE_SIZE] = 0
        array[x:x+img.SHAPE_SIZE, y] += img.PIX_VAL
        return True
    else: return False
