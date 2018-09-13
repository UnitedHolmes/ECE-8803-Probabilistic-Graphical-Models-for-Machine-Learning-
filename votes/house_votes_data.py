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
        votes[k,0] = 0
    elif this_ele[0] == "democrat":
        votes[k,0] = 1
        
    for kk in range(1,17):
        if this_ele[kk] == "n":
            votes[k,kk] = 0
        elif this_ele[kk] == "y":
            votes[k,kk] = 1
        elif this_ele[kk] == "?":
            votes[k,kk] = -1
            
new_votes = votes[:]
k = 0
while k < np.size(new_votes,0):
    if any(new_votes[k,:] == -1):
#        np.delete(new_votes,new_votes[k,:])
        new_votes = np.delete(new_votes,k,0)
    else:
        k = k + 1;
        
num_dem = 0
num_rep = 0
for k in range(0,np.size(new_votes,0)):
    if new_votes[k,0] == 0:
        num_rep = num_rep + 1
    elif new_votes[k,0] == 1:
        num_dem = num_dem + 1;
        
np.save('house_votes',votes,new_votes)