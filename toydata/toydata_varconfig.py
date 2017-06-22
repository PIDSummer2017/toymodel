class test_image:
    def __init__(self):
        self.SHAPE_SIZE = 5
        self.IMSIZE = 28
        self.ALLOWED = [T, T, T, T]
        self.MULTIPLICITIES = [1, 1, 1, 1]

    def parse(self, args):
        for arg in args:
            try:
                if arg.startswith('shape_size='):
                    self.SHAPE_SIZE = arg.replace('shape_size=', '')
                if arg.startswith('image_size='):
                    self.IMSIZE = arg.replace('image_size=','')

                except Exception:
                    print 'argument:', arg, 'not in a valid format! Parsing failed :('

    def __str__(self):
        msg = 'configuration parameters:\n'
        msg += ' Shape Size = %d\n' %self.SHAPE_SIZE
        msg += 'Image Size = %d by %d' %self.IMSIZE

if __name__ == '__main__':
    import sys
    img = test_image()
    img.parse(sus.arg)
    print img
