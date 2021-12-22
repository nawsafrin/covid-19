#!/bin/bash


python twitter_monitor.py --configfile=oxford_config.json --logfile=oxford_short

python twitter_monitor.py --configfile=covaxin_config.json --logfile=covaxin_short

python twitter_monitor.py --configfile=sv_config.json --logfile=sputnikv_short

python twitter_monitor.py --configfile=sinovac_config.json --logfile=sinovac_short
