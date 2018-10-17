#import ROOT,sys,time,os,signal
from larcv import larcv
import sys,time,os,signal
import numpy as np

class larcv_data (object):

   _instance_m={}

   @classmethod
   def exist(cls,name):
      name = str(name)
      return name in cls._instance_m

   def __init__(self):
      self._proc = None
      self._name = ''
      self._verbose = False
      self._read_start_time = None
      self._read_end_time = None
      self._cfg_file = None
      self._batch = -1
      self._image_data = None
      self._image_data_size = None
      self._image_dim = None
      self._label_data = None
      self._label_data_size = None
      self._label_dim = None
      self._multiplicity_data = None
      self._multiplicity_data_size = None
      self._multiplicity_dim = None

      self.time_data_read  = 0
      self.time_data_conv  = 0
      self.time_data_copy  = 0
      self.time_label_conv = 0
      self.time_label_copy = 0
      self.time_multiplicity_conv = 0
      self.time_multiplicity_copy = 0
      self.read_counter = 0

   def image_dim(self):
      return self._image_dim

   def label_dim(self):
      return self._label_dim

   def multiplicity_dim(self):
      return self._multiplicity_dim

   def reset(self):
      if self.is_reading():
         self.next()
      if larcv.ThreadFillerFactory.exist_filler(self._name):
         larcv.ThreadFillerFactory.destroy_filler(self._name)

   def __del__(self):
      self.reset()

   def configure(self,cfg):

      if self._name:
         self.reset()

      if not cfg['filler_name']:
         sys.stderr.write('filler_name is empty!\n')
         raise ValueError

      if self.__class__.exist(cfg['filler_name']):
         sys.stderr.write('filler_name %s already running!' % cfg['filler_name'])
         return

      self._name = cfg['filler_name']         

      self._cfg_file = cfg['filler_cfg']
      if not self._cfg_file or not os.path.isfile(self._cfg_file):
         sys.stderr.write('filler_cfg file does not exist: %s\n' % self._cfg_file)
         sys.stderr.flush()
         raise ValueError

      if 'verbosity' in cfg:
         self._verbose = bool(cfg['verbosity'])

      self.__class__._instance_m[self._name] = self

   def read_next(self,batch):
      batch = int(batch)
      if not batch >0:
         sys.stderr.write('Batch size must be positive integer!\n')
         raise ValueError
      
      if not self._name:
         sys.stderr.write('Filler name unspecified!\n')
         raise ValueError

      if not self._proc:
         self._proc = larcv.ThreadFillerFactory.get_filler(self._name)
         self._proc.configure(self._cfg_file)

      self._read_start_time = time.time()
      self._batch = batch
      self._proc.batch_process(batch)

   def is_reading(self):
      return self._proc.thread_running()
      #return (self._batch > 0)

   def next(self):
      if self._batch <= 0:
         sys.stderr.write('Thread not running...\n')
         raise Exception

      sleep_ctr=0
      while self._proc.thread_running():
         time.sleep(0.005)
         sleep_ctr+=1
         if sleep_ctr%1000 ==0:
            print 'queueing... (waited %-4d sec.)' % (0.005 * sleep_ctr)
            
      #sleep_ctr=0
      #while sleep_ctr<40:
      #   time.sleep(0.125)
      #   sleep_ctr += 1
      #   print self._proc.thread_running(),
      #print

      self._read_end_time = time.time()
      if self.read_counter:
         self.time_data_read += (self._read_end_time - self._read_start_time)

      if self._verbose:
         print
         print 'Data size:',self._proc.data().size()
         print 'Label size:',self._proc.labels().size()
         print 'Multiplicity size:',self._proc.multiplicities().size()
         print 'Batch size:',self._proc.processed_entries().size()
         print 'Total process counter:',self._proc.process_ctr()

      ctime = time.time()
      #np_data = larcv.as_ndarray(self._proc.data()).reshape(self._batch,self._proc.data().size()/self._batch)#.astype(np.float32)
      if self._image_data is None:

         self._image_data = np.array(self._proc.data())
         self._image_data_size = self._image_data.size

         self._label_data = np.array(self._proc.labels())
         self._label_data_size = self._label_data.size

         self._multiplicity_data = np.array(self._proc.multiplicities())
         self._multiplicity_data_size = self._multiplicity_data.size

         self._image_dim = np.array(self._proc.dim(True ))
         self._label_dim = np.array(self._proc.dim(False))
         self._multiplicity_dim = np.array(self._proc.dim(False))
      else:
         self._image_data = self._image_data.reshape(self._image_data.size)
         larcv.copy_array(self._image_data,self._proc.data())

         self._label_data = self._label_data.reshape(self._label_data.size)
         larcv.copy_array(self._label_data,self._proc.labels())

         self._multiplicity_data = self._multiplicity_data.reshape(self._multiplicity_data.size)
         larcv.copy_array(self._multiplicity_data,self._proc.multiplicities())
         
         self._image_dim[0] = self._batch
         self._label_dim[0] = self._batch
         self._multiplicity_dim[0] = self._batch

      if self.read_counter: self.time_data_copy += time.time() - ctime

      ctime = time.time()
      self._image_entry_data_size = self._proc.data().size() / self._batch
      self._image_data = self._image_data.reshape(self._batch,self._image_entry_data_size).astype(np.float32)
      if self.read_counter: self.time_data_conv += time.time() - ctime

      ctime = time.time()
      self._label_entry_data_size = self._proc.labels().size() / self._batch
      self._label_data = self._label_data.reshape(self._batch,self._label_entry_data_size).astype(np.float32)
      if self.read_counter: self.time_label_conv += time.time() - ctime

      ctime = time.time()
      self._multiplicity_entry_data_size = self._proc.multiplicities().size() / self._batch
      self._multiplicity_data = self._multiplicity_data.reshape(self._batch,self._multiplicity_entry_data_size).astype(np.float32)
      if self.read_counter: self.time_multiplicity_conv += time.time() - ctime

      self._batch = -1
      self.read_counter += 1
      return (self._image_data,self._label_data, self._multiplicity_data)

def sig_kill(signal,frame):
   print '\033[95mSIGINT detected.\033[00m Finishing the program gracefully.'
   for name,ptr in larcv_data._instance_m.iteritems():
      print 'Terminating filler:',name
      ptr.reset()

signal.signal(signal.SIGINT,  sig_kill)



