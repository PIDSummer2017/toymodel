class toy_config:

    def __init__(self):

        self.NUM_CLASS        = 5
        self.TRAIN_BATCH_SIZE = 100
        self.TEST_BATCH_SIZE  = 100
        self.ANA_BATCH_SIZE   = 100
        self.TRAIN_ITERATIONS = 1000
        self.LOGDIR           = 'logs'
        self.ARCHITECTURE     = 'lenet'
        self.BAD_LABEL        = False

        self.TEST_BATCH_SIZE  = 100
        self.ANA_BATCH_SIZE   = 100
        self.TRAIN_ITERATIONS = 1000
        self.LOGDIR           = 'logs'
       # self.ARCHITECTURE     = 'multi_lenet'

        self.MULTI_LABEL      = False
        self.DEBUG            = 0
        self.ANA_FILE         = ' '
        self.LOAD_FILE        = True
        self.ANA_SAVE         = True
        self.TRAIN_SAVE       = True
    
    def parse(self,argv_v):

        cfg_file=None
        for argv in argv_v:
            if argv.endswith('.cfg'):
                params=open(argv,'r').read().split()
                return self.parse(params)

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
                elif argv.startswith('bad_label='):
                    self.BAD_LABEL = int(argv.replace('bad_label=',''))
                elif argv.startswith('multi_label='):
                    self.MULTI_LABEL = int(argv.replace('multi_label=',''))
                elif argv.startswith('ana_file='):
                    self.ANA_FILE = argv.replace('ana_file=','')
                elif argv.startswith('load_file='):
                    self.LOAD_FILE = argv.replace('load_file=','')
                elif argv.startswith('ana_save='):
                    self.SAVE_FILE = argv.replace('ana_save=','')
                elif argv.startswith('train_save='):
                    self.SAVE_FILE = argv.replace('train_save=','') 
 
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
        msg += '    bad label          = %d\n' % self.BAD_LABEL
        msg += '    multi label        = %d\n' % self.MULTI_LABEL
        return msg

if __name__ == '__main__':
    import sys
    cfg = toy_config()
    cfg.parse(sys.argv)
    print cfg
