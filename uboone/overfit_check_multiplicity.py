# Basic imports
import os,sys,time,glob
from toytrain import config
import numpy as np
import tensorflow as tf

ptypes=['eminus','gamma','muon','pion','proton']

#
# Utility functions
#
import numpy as np
# Integer rounder
def time_round(num,digits):
  return float( int(num * np.power(10,digits)) / float(np.power(10,digits)) )

class truth_info:
  def __init__(self):
    self.multi_all = 0
    self.multi_neutron = 0
    self.multi_sum = 0
    self.multi_v = [0]*5
    #self.eminus_multi=0
    #self.gamma_multi=0
    #self.muon_multi=0
    #self.pion_multi=0
    #self.proton_multi=0
    self.max_energy_v = [-1.]*5
    self.min_energy_v = [-1.]*5
    self.energy_sum = 0.
    self.dcosz_v = [-2]*5
    self.open_angle = -1

def get_truth_info(roi_chain,entry):
  mass = [0.511,0.,105.6,139.6,938.28]
  roi_chain.GetEntry(entry)
  roi_v = roi_chain.partroi_segment_branch.ROIArray()
  res = truth_info()
  two_part_index = []
  res.multi_v = [0]*5
  for idx in xrange(roi_v.size()):
    roi = roi_v[idx]
    energy = roi.EnergyInit()
    index = -1
    if   np.abs(roi.PdgCode()) == 11: 
      index = 0
      #res.eminus_multi+=1
    elif np.abs(roi.PdgCode()) == 22: 
      index = 1
      #res.gamma_multi+=1
    elif np.abs(roi.PdgCode()) == 13: 
      index = 2
      #res.muon_multi+=1
    elif np.abs(roi.PdgCode()) == 211: 
      index = 3
      #res.pion_multi+=1
    elif np.abs(roi.PdgCode()) == 2212: 
      index = 4
      #res.proton_multi+=1
    res.multi_all +=1
    if np.abs(roi.PdgCode()) == 2112: res.multi_neutron += 1
    
    if index<0: continue

    two_part_index.append(idx)
    res.multi_v[index] += 1
    energy -= mass[index]
    res.energy_sum += energy
    if res.max_energy_v[index] < 0 or res.max_energy_v[index] < energy:
      res.max_energy_v[index] = energy
      res.dcosz_v[index] = roi.Pz() / np.sqrt(np.power(roi.Px(),2) + np.power(roi.Py(),2) +np.power(roi.Pz(),2))
    if res.min_energy_v[index] < 0 or res.min_energy_v[index] > energy:
      res.min_energy_v[index] = energy

  for v in res.multi_v:
    res.multi_sum += v

  if len(two_part_index) == 2:
    part1 = roi_v[two_part_index[0]]
    part2 = roi_v[two_part_index[1]]
    mag1 = np.sqrt(np.power(part1.Px(),2)+np.power(part1.Py(),2)+np.power(part1.Pz(),2))
    mag2 = np.sqrt(np.power(part2.Px(),2)+np.power(part2.Py(),2)+np.power(part2.Pz(),2))
    res.open_angle = part1.Px() * part2.Px() + part1.Py() * part2.Py() + part1.Pz() * part2.Pz()
    res.open_angle = np.arccos(res.open_angle / (mag1 * mag2)) / np.pi * 180.
  else:
    res.open_angle = -1

  return res

def main():

  # Load configuration and check if it's good

  cfg = config()
  if not cfg.parse(sys.argv) or not cfg.sanity_check():
    sys.exit(1)

  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

  #os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.PLANE+2)
  os.environ["CUDA_VISIBLE_DEVICES"]="0"
  # Print configuration
  print '\033[95mConfiguration\033[00m'
  print cfg
  time.sleep(0.5)

  # Import more libraries (after configuration is validated)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from dataloader import larcv_data

  #########################
  # main part starts here #
  #########################

  #
  # Step 0: configure IO
  #

  # Instantiate and configure
  if not cfg.FILLER_CONFIG:
    print 'Must provide larcv data filler configuration file!'
    sys.exit(1)

  proc = larcv_data()
  filler_cfg = {'filler_name': 'DataFiller', 
                'verbosity':0, 
                'filler_cfg':cfg.FILLER_CONFIG}
  proc.configure(filler_cfg)
  # Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
  proc.read_next(cfg.BATCH_SIZE)
  # Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
  proc.next()
  # Retrieve image/label dimensions
  image_dim = proc.image_dim()
  label_dim = proc.label_dim()

  #
  # Step 1: prepare truth information handle
  #
  from larcv import larcv
  from ROOT import TChain
  filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
  roi_chain = TChain("partroi_segment_tree")
  for fname in filler.pd().io().file_list():
    roi_chain.AddFile(fname)
  filler.set_next_index(5861)
  # Immediately start the thread for later IO
  proc.read_next(cfg.BATCH_SIZE)

  #
  # Step 2: Build network
  #

  # Set input data and label for training
  data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')
  label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')
  data_tensor_2d = tf.reshape(data_tensor, [-1,image_dim[2],image_dim[3],1])

  # Call network build function (then we add more train-specific layers)
  net = None
  #cmd = 'from toynet import toy_%s;net=toy_%s.build(data_tensor_2d,cfg.NUM_CLASS)' % (cfg.ARCHITECTURE,cfg.ARCHITECTURE)
  cmd = 'from toynet import toy_%s;net =toy_%s.build(data_tensor_2d,cfg.NUM_CLASS,keep_prob = 1.0)' % (cfg.ARCHITECTURE,cfg.ARCHITECTURE)
  exec(cmd)

  # Define accuracy
  with tf.name_scope('sigmoid'):
    sigmoid = tf.nn.sigmoid(net)

  #
  # Step 3: Configure global process (session, summary, etc.)
  #
  # Create a session
  sess = tf.InteractiveSession()
  # Initialize variables
  sess.run(tf.global_variables_initializer())
  # Override variables if wished
  reader=tf.train.Saver()
  
  list_of_files = glob.glob('plane%straining/multiplicity/*'%(cfg.PLANE))
  latest_file = max(list_of_files, key=os.path.getctime)
  weight_file_path = latest_file.split(".")[0]
  weight_file_name =  latest_file.split(".")[0].split("/")[2]

  print '========>>>>',weight_file_path

  reader.restore(sess,weight_file_path)
  # Analysis csv file
  #weight_file_name = cfg.LOAD_FILE.split('/')[-1]
  filler_file_name = cfg.FILLER_CONFIG.split('/')[-1].replace('.cfg','')
  fout = open('test_csv/plane%s/multiplicity/%s.%s.csv' % (cfg.PLANE,weight_file_name,filler_file_name),'w')
  print '===============>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
  print 'test_csv/plane%s/multiplicity/%s.%s.csv' % (cfg.PLANE,weight_file_name,filler_file_name) 
  fout.write('entry,label0,label1,label2,label3,label4')
  for idx in xrange(cfg.MULTIPLICITY_CLASS):
    idy=int(idx/5.)
    ptype=ptypes[idy]
    mul=idx%5
    fout.write(',%s_multi_%i' % (ptype,mul))
  for idx in xrange(cfg.NUM_CLASS):
    fout.write(',max_energy%02d' % idx)
  for idx in xrange(cfg.NUM_CLASS):
    fout.write(',min_energy%02d' % idx)
  for idx in xrange(cfg.NUM_CLASS):
    fout.write(',dcosz%02d' % idx)
  fout.write(',multi_all,multi_neutron,multi_sum,energy_sum,open_angle')
  fout.write('\n')

  # Run training loop
  entry_number_v = [0] * cfg.BATCH_SIZE
  for i in range(cfg.ITERATIONS):
    # Report the progress
    sys.stdout.write('Processing %d/%d\r' % (i,cfg.ITERATIONS))
    sys.stdout.flush()
    # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
    data,label,multiplicity = proc.next()
    processed_entries = filler.processed_entries()
    for entry in xrange(processed_entries.size()):
      entry_number_v[entry] = processed_entries[entry]
    # Run loss & train step
    score_vv = sess.run(sigmoid,feed_dict={data_tensor: data})
                
    for res_idx,score_v in enumerate(score_vv):
      entry = entry_number_v[res_idx]
      fout.write('%d' % entry)

      mcinfo = get_truth_info(roi_chain, entry)
      for v in mcinfo.multi_v:
        fout.write(',%d' % int(v))
      for score in score_v:
        fout.write(',%g' % score)
      for v in mcinfo.max_energy_v:
        fout.write(',%g' % v)
      for v in mcinfo.min_energy_v:
        fout.write(',%g' % v)
      for v in mcinfo.dcosz_v:
        fout.write(',%g' % v)
      fout.write(',%d' % mcinfo.multi_all)
      fout.write(',%d' % mcinfo.multi_neutron)
      fout.write(',%d' % mcinfo.multi_sum)
      fout.write(',%g' % mcinfo.energy_sum)
      fout.write(',%g' % mcinfo.open_angle)
      fout.write('\n')
    # Start IO thread for the next batch 

    proc.read_next(cfg.BATCH_SIZE)
  fout.close()
  print
  print 'Done'

if __name__ == '__main__':
  from choose_gpu import pick_gpu
  GPUMEM=10000
  GPUID=pick_gpu(GPUMEM,caffe_gpuid=True)
  print GPUID
  if GPUID < 0:
    sys.stderr.write('No available GPU with memory %d\n' % GPUMEM)
    sys.exit(1)
    #with tf.device('/gpu:%d' % GPUID):
  main()
