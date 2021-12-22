#!/bin/bash


# python main.py --input=../data/health/wash/wash -n Hand-Wash
# python main.py --input=../data/health/mask/mask -n Mask
# python main.py --input=../data/health/travel/travel -n Travel
# echo "social_distancing"
# python main.py --input=../data/health/social_distancing/social_distancing -n Social-Distancing

# echo "social_gathering"

# python main.py --input=../data/health/social_gathering/social_gathering -n Social-Gathering


echo "Healthy Habits"
python main_location.py --input=../data/health/health_unique -n Healthy-Habits

# echo "Oxford-AstraZeneca"

# python main_location.py --input=../data/vaccine/oxford/oxford -n Oxford-AstraZeneca 

# echo "JohnsonAndJohnson"
# python main_location.py --input=../data/vaccine/jj/jj -n JohnsonAndJohnson
# echo "Sinovac"
# python main_location.py --input=../data/vaccine/sinovac/sinovac -n Sinovac

# echo "SputnikV"
# python main_location.py --input=../data/vaccine/sputnikv/sputnikv -n SputnikV
# echo "Covaxin"
# python main_location.py --input=../data/vaccine/covaxin/covaxin-c -n Covaxin



# echo "Moderna"


# python main_location.py --input=../data/vaccine/moderna/moderna -n Moderna


# echo "Pfizer"
# python main_location.py --input=../data/vaccine/pfizer/pfizer -n Pfizer 
