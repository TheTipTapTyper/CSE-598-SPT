"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 4/17/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""
import numpy as np  



def angle_between_vecs(v1, v2):
    """ returns the angle between two vectors (in randians)
    Reference: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def distance_between_vecs(v1, v2):
    return np.linalg.norm(v1 - v2)