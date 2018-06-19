#!/bin/sh
SHELL=/bin/bash

source /home/dayajun/.bash_profile
source /home/dayajun/setup.sh
cd toymodel/uboone
python overfit_check_pid.py pid_multiclass_ana_2.cfg
python overfit_store_pid.py pid_multiclass_ana_2.cfg
#python overfit_check_pid.py pid_multiclass_ana_1.cfg
#python overfit_store_pid.py pid_multiclass_ana_1.cfg
python monitor.py pid_multiclass_ana.cfg pid
