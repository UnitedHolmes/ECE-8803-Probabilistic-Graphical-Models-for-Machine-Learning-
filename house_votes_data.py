# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:32:03 2018

@author: yzhu94
"""
import numpy as np

file = open("house-votes-84.data","r")
lines = file.readlines()
#votes_democrats = np.zeros([len(lines),16])
#votes_republican = np.zeros([len(lines),16])
# For matrix votes, first column is party
# 0 for republican, 1 for democrat
votes = np.zeros([len(lines),17])

for k in range(0,len(lines)):
    this_line = lines[k]
    this_ele = this_line.split(",")
    if this_ele[0] == "republican":
        votes[k,0] = -1
    elif this_ele[0] == "democrat":
        votes[k,0] = 1
        
    for kk in range(1,17):
        if this_ele[kk] == "n" or this_ele[kk] == "n\n":
            votes[k,kk] = -1
        elif this_ele[kk] == "y" or this_ele[kk] == "y\n":
            votes[k,kk] = 1
        elif this_ele[kk] == "?" or this_ele[kk] == "?\n":
            votes[k,kk] = 0
            
clean_votes = votes[:]
k = 0
while k < np.size(clean_votes,0):
    if any(clean_votes[k,:] == 0):
        clean_votes = np.delete(clean_votes,k,0)
    else:
        k = k + 1;
        
num_dem = 0
num_rep = 0
for k in range(0,np.size(clean_votes,0)):
    if clean_votes[k,0] == -1:
        num_rep = num_rep + 1
    elif clean_votes[k,0] == 1:
        num_dem = num_dem + 1;
        
np.save('house_votes_total',votes)
np.save('house_votes_clean',clean_votes)