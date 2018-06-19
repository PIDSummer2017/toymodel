#!/bin/sh
SHELL=/bin/bash

source /home/dayajun/.bash_profile
source /home/dayajun/setup.sh
cd toymodel/uboone
python overfit_check_multiplicity.py pid_multiplicity_ana.cfg
python overfit_store_multiplicity.py pid_multiplicity_ana.cfg

python monitor.py pid_multiplicity_ana.cfg multiplicity
