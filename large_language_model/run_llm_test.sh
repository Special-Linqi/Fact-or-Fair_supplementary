#!/bin/bash

# Navigate to objective_test, run scripts, then return
cd objective_test || exit
python obj_test.py
python obj_analysis.py
cd ..

# Navigate to subjective_test, run scripts, then return
cd subjective_test || exit
python subj_test.py
python subj_analysis.py
cd ..
