class test_image:
    def __init__(self):
        self.SHAPE_SIZE = 5
        self.PIX_VAL = 180
        self.IMSIZE = 28
        self.ALLOWED = [True, True, True, True]
        self.MULTIPLICITIES = [1, 1, 1, 1]

    def parse(self, args):
        for arg in args:
            try:
                if arg.startswith('shape_size='):
                    self.SHAPE_SIZE = arg.replace('shape_size=', '')
                if arg.startswith('image_size='):
                    self.IMSIZE = arg.replace('image_size=','')
                if arg.startswith('pixel_val='):
                    self.PIX_VAL = arg.replace('pixel_val=','')
                if arg.startswith('allowed_shapes='):
                    self.ALLOWED = arg.replace('allowed_shapes=','')
                if arg.startswith('multiplicities='):
                    self.MULTIPLICITIES = arg.replace('multiplicities=','')
            except Exception:
                print 'argument:', arg, 'not in a valid format! Parsing failed :('
=======
                print('argument:', arg, 'not in a valid format! Parsing failed :(')

   # def __str__(self):
  #      msg = 'configuration parameters:\n'
 #       msg += ' Shape Size = %d\n' %self.SHAPE_SIZE
#        msg += 'Image Size = %d' %self.IMSIZE

if __name__ == '__main__':
    import sys
    img = test_image()
    img.parse(sus.arg)
    print img
