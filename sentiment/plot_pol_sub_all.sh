#!/bin/bash

# echo "hygiene"
# python plot_pol_sub.py --input=../data/health/wash/wash -n Hand-Wash
# echo "mask"
# python plot_pol_sub.py --input=../data/health/mask/mask -n Mask
# echo "travel"
# python plot_pol_sub.py --input=../data/health/travel/travel -n Travel
# echo "social_distancing"
# python plot_pol_sub.py --input=../data/health/social_distancing/social_distancing -n Social-Distancing

# echo "social_gathering"

# python plot_pol_sub.py --input=../data/health/social_gathering/social_gathering -n Social-Gathering






echo "Oxford"

python plot_pol_sub.py --input=../data/vaccine/oxford/oxford -n Oxford-AstraZeneca 



echo "Moderna"
python plot_pol_sub.py --input=../data/vaccine/moderna/moderna -n Moderna

echo "JohnsonAndJohnson"
python plot_pol_sub.py --input=../data/vaccine/jj/jj -n JohnsonAndJohnson
echo "Sinovac"
python plot_pol_sub.py --input=../data/vaccine/sinovac/sinovac -n Sinovac

echo "SputnikV"
python plot_pol_sub.py --input=../data/vaccine/sputnikv/sputnikv -n SputnikV
echo "Covaxin"
python plot_pol_sub.py --input=../data/vaccine/covaxin/covaxin-c -n Covaxin
echo "Pfizer"
python plot_pol_sub.py --input=../data/vaccine/pfizer/pfizer -n Pfizer 
