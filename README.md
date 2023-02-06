# Trajectories Clustering
Use the Ramer-Douglas-Peucker (RDP) algorithm to reduce the number of points in a trajectory from EDVAM,
and use traclus to do clustering task. The performance is bad and still need to **find better hyper parameters**.

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

clustering and generate stand trajectory

```python
python partition_cluster.py
```

smooth the trajectory.

```python
bessel_interpolation.py
```

## Main File

+ main.py: RDP algorithm.

+ main_draw.py: draw the trajectories of the data to show the diffence before and after RDP.

+ cluster.py: simple cluster the trajectory without partition.

+ partition_cluster.py: partition and clutering of trajectory

## Reference

https://github.com/apolcyn/traclus_impl