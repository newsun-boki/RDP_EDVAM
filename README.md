# Trajectories Clustering
Use the Ramer-Douglas-Peucker (RDP) algorithm to reduce the number of points in a trajectory from EDVAM,
and do clustering. The performance is not good though and still need to find better hyper parameters.

## Requirments

+ python 3.7
+ pandas
+ matplotlib
+ rdp

## Quick Start

```python
pip install -r requirements.txt
```

generate RDP files

```python
python main.py
```

draw trajectories.

```python
python main_draw.py
```

## Main File

+ main.py: RDP algorithm.

+ main_draw.py: draw the trajectories of the data to show the diffence before and after RDP.

+ cluster.py: simple cluster the trajectory without partition.

+ partition_cluster.py: partition and clutering of trajectory

## Reference

https://github.com/apolcyn/traclus_impl