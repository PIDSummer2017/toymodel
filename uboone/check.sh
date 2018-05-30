#!/bin/sh
SHELL=/bin/bash

source /home/dayajun/.bash_profile
source /home/dayajun/setup.sh
cd toymodel/uboone
python overfit_check.py pid_multiclass_ana.cfg
python overfit_store.py pid_multiclass_ana.cfg
python monitor.py pid_multiclass_ana.cfg
