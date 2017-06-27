
class toy_config:

    def __init__(self):
        self.NUM_CLASS        = 8
        self.TRAIN_BATCH_SIZE = 100
        self.TEST_BATCH_SIZE  = 1000
        self.ANA_BATCH_SIZE   = 0
        self.TRAIN_ITERATIONS = 5000
        self.LOGDIR           = 'multlogs'
        self.ARCHITECTURE     = 'lenet'
        self.DEBUG            = 0

    def parse(self,argv_v):

        for argv in argv_v:
            try:
                if   argv.startswith('num_class='):
                    self.NUM_CLASS = int(argv.replace('num_class=',''))
                elif argv.startswith('train_batch='):
                    self.TRAIN_BATCH_SIZE = int(argv.replace('train_batch=',''))
                elif argv.startswith('test_batch='):
                    self.TEST_BATCH_SIZE = int(argv.replace('test_batch=',''))
                elif argv.startswith('ana_batch='):
                    self.ANA_BATCH_SIZE = int(argv.replace('ana_batch=',''))
                elif argv.startswith('iterations='):
                    self.TRAIN_ITERATIONS = int(argv.replace('iterations=',''))
                elif argv.startswith('logdir='):
                    self.LOGDIR = argv.replace('logdir=','')
                elif argv.startswith('arch='):
                    self.ARCHITECTURE = argv.replace('arch=','')
                elif argv.startswith('debug='):
                    self.DEBUG = int(argv.replace('debug=',''))

            except Exception:
                print 'argument:',argv,'not in a valid format (parsing failed!)'
                return False
        return True
        
    def __str__(self):
        msg  = 'Configuration parameters:\n'
        msg += '    class count        = %d\n' % self.NUM_CLASS
        msg += '    batch size (train) = %d\n' % self.TRAIN_BATCH_SIZE
        msg += '    batch size (test)  = %d\n' % self.TEST_BATCH_SIZE
        msg += '    batch size (ana)   = %d\n' % self.ANA_BATCH_SIZE
        msg += '    train iterations   = %d\n' % self.TRAIN_ITERATIONS
        msg += '    log directory      = %s\n' % self.LOGDIR
        msg += '    architecture       = %s\n' % self.ARCHITECTURE
        msg += '    debug mode         = %d\n' % self.DEBUG
        return msg

if __name__ == '__main__':
    import sys
    cfg = toy_config()
    cfg.parse(sys.argv)
    print cfg
