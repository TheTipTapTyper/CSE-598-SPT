"""
Author: Ryan Markson
Email: ryanmarkson1@gmail.com
Created: 3/18/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""

import numpy as np  
from vec_math import angle_between_vecs, distance_between_vecs
import lm_extractor as lme


MAX_KNEE_BAR_DIST = .2
MIN_NECK_ANGLE = 165 * np.pi / 180
MAX_FEET_SHOULDER_WIDTH_DIFF = .2
MAX_BAR_ANGLE = 15 * np.pi / 180



## Side View Metrics ###

def knee_bar_dist(sv_lm_array):
    """ Calculates horizontal distance between the middle of the bar and the knee.
    Input: 8x3 sideview landmark array
    Returns: horizontal distance between barbell (hands) and knee
    """
    knee = sv_lm_array[lme.KNEE, lme.X_IDX]
    bar = sv_lm_array[lme.INDEX, lme.X_IDX]
    return distance_between_vecs(knee, bar)

def knee_shoulder_dist(sv_lm_array):
    """ Calculates horizontal distance between the knee and the shoulder.
    Input: 8x3 sideview landmark array
    Returns: horizontal distance between the knee and the shoulder
    """
    knee = sv_lm_array[lme.KNEE, lme.X_IDX]
    shoulder = sv_lm_array[lme.SHOULDER, lme.X_IDX]
    return distance_between_vecs(knee, shoulder)

def knee_angle(sv_lm_array):
    """ Calculates the angle of the knee joint.
    Input: 8x3 sideview landmark array
    Returns: angle of the knee joint in radians
    """
    knee = sv_lm_array[lme.KNEE, [lme.X_IDX, lme.Y_IDX]]
    ankle = sv_lm_array[lme.ANKLE, [lme.X_IDX, lme.Y_IDX]] - knee
    hip = sv_lm_array[lme.HIP, [lme.X_IDX, lme.Y_IDX]] - knee
    return angle_between_vecs(ankle, hip)

def neck_angle(sv_lm_array):
    """ Calculates the angle of the neck joint.
    Input: 8x3 sideview landmark array
    Returns: angle of the neck joint in radians
    """
    shoulder = sv_lm_array[lme.SHOULDER, [lme.X_IDX, lme.Y_IDX]]
    ear = sv_lm_array[lme.EAR, [lme.X_IDX, lme.Y_IDX]] - shoulder
    hip = sv_lm_array[lme.HIP, [lme.X_IDX, lme.Y_IDX]] - shoulder
    return angle_between_vecs(ear, hip)


## Side View Heuristics ##

def knee_bar_heuristic(sv_lm_array):
    """ Determines whether the knee-to-bar distance is within the desired bounds.
    Input: 8x3 sideview landmark array
    Returns: bool
    """
    dist = knee_bar_dist(sv_lm_array)
    return dist < MAX_KNEE_BAR_DIST

def knee_shoulder_heuristic(sv_lm_array):
    """ Determines whether the knee-to-shoulder distance is within the desired
    bounds, as a function of the knee angle. Knee angle is used to approximate
    the phase of the deadlift.
    Input: 8x3 sideview landmark array
    Returns: bool
    """
    pass

def neck_angle_heuristic(sv_lm_array):
    """ Determines whether the neck angle is within the desired bounds.
    Input: 8x3 sideview landmark array
    Returns: bool
    """
    angle = neck_angle(sv_lm_array)
    print(angle * 180 / np.pi)
    return angle > MIN_NECK_ANGLE

## Front View Metrics ##

def feet_width(lm_array):
    """ Calculate the distance between the feet in the xz-plane.
    Input: 33x4 landmark array
    Returns: distance between feet
    """
    idxs = lme.mp.solutions.pose.PoseLandmark
    left_ankle = lm_array[idxs.LEFT_ANKLE, [lme.X_IDX, lme.Z_IDX]]
    right_ankle = lm_array[idxs.RIGHT_ANKLE, [lme.Y_IDX, lme.Z_IDX]]
    return distance_between_vecs(left_ankle, right_ankle)

def shoulder_width(lm_array):
    """ Calculate the distance between the shoulders in the xz-plane.
    Input: 33x4 landmark array
    Returns: distance between shoulders
    """
    idxs = lme.mp.solutions.pose.PoseLandmark
    left_shoulder = lm_array[idxs.LEFT_SHOULDER, [lme.X_IDX, lme.Z_IDX]]
    right_shoulder = lm_array[idxs.RIGHT_SHOULDER, [lme.Y_IDX, lme.Z_IDX]]
    return distance_between_vecs(left_shoulder, right_shoulder)

def bar_angle(lm_array):
    """ Calculates the angle between the bar and the x axis.
    Input: 33x4 landmark array
    Returns: angle between bar and horizontal in radians
    """
    idxs = lme.mp.solutions.pose.PoseLandmark
    left_hand = lm_array[idxs.LEFT_INDEX, [lme.X_IDX, lme.Y_IDX]]
    right_hand = lm_array[idxs.RIGHT_INDEX, [lme.X_IDX, lme.Y_IDX]]
    bar_mid = np.mean([left_hand, right_hand], axis=0)
    bar_direction = left_hand - bar_mid
    return angle_between_vecs(bar_direction, np.array([1, 0]))

## Front View Heuristics ##

def feet_width_heuristic(lm_array):
    """ Determines whether the feet are close enough to shoulder width apart.
    Input: 33x4 landmark array
    Returns: bool
    """
    diff = np.abs(feet_width(lm_array) - shoulder_width(lm_array))
    return diff < MAX_FEET_SHOULDER_WIDTH_DIFF

def bar_angle_heuristic(lm_array):
    """ Determines whether the bar angle is within the desired bounds.
    Input: 33x4 landmark array
    Returns: bool
    """
    angle = bar_angle(lm_array)
    return angle < MAX_BAR_ANGLE



                

          
           

                    
                
            












            
            
            
            

        
    
    
 
        
 
        
        
        
        
        
    
   
