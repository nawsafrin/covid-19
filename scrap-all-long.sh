#!/bin/bash

python twitter_monitor.py --configfile=./config-long/oxford_config.json --logfile=oxford_long

python twitter_monitor.py --configfile=./config-long/sinovac_config.json --logfile=sinovac_long

python twitter_monitor.py --configfile=./config-long/covaxin_config.json --logfile=covaxin_long

python twitter_monitor.py --configfile=./config-long/sv_config.json --logfile=sputnikv_long

python twitter_monitor.py --configfile=./config-long/jj_config.json --logfile=jj_long

python twitter_monitor.py --configfile=./config-long/mo_config.json --logfile=moderna_long

python twitter_monitor.py --configfile=./config-long/pfizer_config.json --logfile=pfizer_long


