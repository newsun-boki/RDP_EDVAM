# RDP

Use the Ramer-Douglas-Peucker (RDP) algorithm to reduce the number of points in a trajectory from EDVAM.

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

## File Structure

+ main.py: RDP algorithm.

+ main_draw: draw the trajectories of the data to show the diffence before and after RDP.
