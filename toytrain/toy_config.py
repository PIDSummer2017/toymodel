
import os, sys, toynet, glob

class config:

    def __init__(self):
        self.DEBUG      = False
        self.NUM_CLASS  = 5
        self.ITERATIONS = 1000
        self.BATCH_SIZE = 100
        self.SAVE_ITERATION = 100
        self.LOGDIR         = 'logs'
        self.ARCHITECTURE   = 'lenet'
        self.LOAD_FILE      = ''
        self.AVOID_LOAD_PARAMS = ''
        self.FILLER_CONFIG = ''
        self.TRAIN_FILLER_CONFIG = ''
        self.TEST_FILLER_CONFIG = ''
        self.GPU_INDEX = ''
        self.PLANE = ''
        self.LATEST_DIR=''
        self.LATEST_FILE=''
    def parse(self,argv_v):

        cfg_file=None
        if len(argv_v) == 2 and argv_v[1].endswith('.cfg'):
            params=open(argv_v[1],'r').read().split()
            return self.parse(params)

        # Get the latest weights

        for argv in argv_v:
            try:
                if   argv.startswith('num_class='):
                    self.NUM_CLASS = int(argv.replace('num_class=',''))
                elif argv.startswith('batch='):
                    self.BATCH_SIZE = int(argv.replace('batch=',''))
                elif argv.startswith('iterations='):
                    self.ITERATIONS = int(argv.replace('iterations=',''))
                elif argv.startswith('logdir='):
                    self.LOGDIR = argv.replace('logdir=','')
                elif argv.startswith('arch='):
                    self.ARCHITECTURE = argv.replace('arch=','')
                elif argv.startswith('debug='):
                    self.DEBUG = int(argv.replace('debug=',''))
                elif argv.startswith('load_file='):
                    self.LOAD_FILE = argv.replace('load_file=','')
                elif argv.startswith('save_iteration='):
                    self.SAVE_ITERATION = int(argv.replace('save_iteration=','') )
                elif argv.startswith('avoid_params='):
                    self.AVOID_LOAD_PARAMS = argv.replace('avoid_params=','')
                elif argv.startswith('filler='):
                    self.FILLER_CONFIG = argv.replace('filler=','')
                elif argv.startswith('train_filler='):
                    self.TRAIN_FILLER_CONFIG = argv.replace('train_filler=','')
                elif argv.startswith('test_filler='):
                    self.TEST_FILLER_CONFIG = argv.replace('test_filler=','')
                elif argv.startswith('gpu_index='):
                    self.GPU_INDEX = int(argv.replace('gpu_index=',''))
                elif argv.startswith('plane='):
                    self.PLANE = int(argv.replace('plane=',''))
                elif argv.startswith('cross_check_dir='):
                    self.LATEST_DIR = argv.replace('cross_check_dir=','')
                if(self.LATEST_DIR):
                    dir_files=self.LATEST_DIR+'*'
                    list_of_files = glob.glob(dir_files)
                    latest_file = max(list_of_files, key=os.path.getctime)
                    pid_file = os.path.splitext(os.path.basename(latest_file))[0]
                    pid_path_file = os.path.join(self.LATEST_DIR, pid_file)
                    self.LATEST_FILE=pid_path_file

            except Exception:
                print 'argument:',argv,'not in a valid format (parsing failed!)'
                return False
        return True

    def ask_binary(self,comment):
        user_input=None
        while user_input is None:
            sys.stdout.write('%s [y/n]:' % comment)
            sys.stdout.flush()
            user_input = sys.stdin.readline().rstrip('\n')
            if not user_input.lower() in ['y','n','yes','no']:
                print 'Unsupported answer:',user_input
                user_input=None
                continue
            return user_input in ['y','yes']

    def check_log(self):
        # Check if log directory already exists
        if not os.path.isdir(self.LOGDIR): 
            print '[NOTICE] Creating a log directory:',self.LOGDIR
            os.mkdir(self.LOGDIR)
            return os.path.isdir(self.LOGDIR)

        print '[WARNING] Log directory already present:',self.LOGDIR
        
        os.system('rm -rf %s' % self.LOGDIR)                                                                                                                                       
        return True
        '''
        rmdir = self.ask_binary('Remove and proceed?')
        if rmdir:
            os.system('rm -rf %s' % self.LOGDIR)
            return True
        else:
            return self.ask_binary('Proceed anyway?')
        '''
    def sanity_check(self):
        # log directory duplication
        if not self.check_log():
            return False
        # filler check
        if self.TRAIN_FILLER_CONFIG:
            if not os.path.isfile(self.TRAIN_FILLER_CONFIG):
                self.TRAIN_FILLER_CONFIG = '%s/uboone/%s' % (os.environ['TOYMODEL_DIR'],self.TRAIN_FILLER_CONFIG)
                if not os.path.isfile(self.TRAIN_FILLER_CONFIG):
                    print 'LArCV data filler config file does not exist!'
                    return False
        # network availability
        try:
            cmd = 'from toynet import toy_%s' % self.ARCHITECTURE
            exec(cmd)
        except Exception:
            print 'Architecture',self.ARCHITECTURE,'is not available...'
            return False

        return True
        
    def __str__(self):
        msg  = 'Configuration parameters:\n'
        msg += '    class count        = %d\n' % self.NUM_CLASS
        msg += '    batch size         = %d\n' % self.BATCH_SIZE
        msg += '    iterations         = %d\n' % self.ITERATIONS
        msg += '    log directory      = %s\n' % self.LOGDIR
        msg += '    architecture       = %s\n' % self.ARCHITECTURE
        msg += '    debug mode         = %d\n' % self.DEBUG
        msg += '    load file?         = %s\n' % self.LOAD_FILE
        msg += '    save per iteration = %s\n' % self.SAVE_ITERATION
        msg += '    avoid params       = %s\n' % self.AVOID_LOAD_PARAMS
        msg += '    gpu index          = %s\n' % self.GPU_INDEX
        msg += '    plane              = %s\n' % self.PLANE
        msg += '    cross check dir    = %s\n' % self.LATEST_DIR
        msg += '    cross check file   = %s\n' % self.LATEST_FILE
        return msg

if __name__ == '__main__':
    import sys
    cfg = toy_config()
    cfg.parse(sys.argv)
    print cfg
