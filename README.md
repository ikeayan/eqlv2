# The Equalization Losses for Long-tailed Object Detection and Instance Segmentation

This branch is based on the official implementation CVPR 2021 paper: **Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection** and CVPR 2020 paper: **Equalization loss for long-tailed object recognition**

A discount factor is added to the eqlv2.py file. To run this branch, use the same way you run the main branch. 

The default discount factor is 0.965. To change it, you have to go into the eqlv2.py file to directly change the varible discount in the __init__ function. When the discount factor = 1, eqlv2.py behaves the same as the code in main branch. 