#!/bin/bash

python twitter_monitor.py --configfile=./config-health/sd_config.json --logfile=social_distancing

python twitter_monitor.py --configfile=./config-health/sg_config.json --logfile=social_gathering

python twitter_monitor.py --configfile=./config-health/travel_config.json --logfile=travel

python twitter_monitor.py --configfile=./config-health/mask_config.json --logfile=mask

python twitter_monitor.py --configfile=./config-health/wash_config.json --logfile=wash




