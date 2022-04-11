"""
Author: Ryan Markson
Email: ryanmarkson1@gmail.com
Created: 3/18/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""

import numpy as np  
from numpy.linalg import norm 
import math


def distance(x,y):
    """Returns the absolute distance between two points. """
    dist = norm(x-y, 2)
    return dist
    
def angle(Vec1, Vec2):
    """Returns the angle between two vectors. """
    
    cosine_angle = np.dot(Vec1, Vec2)/(np.dot(Vec1,Vec1)*np.dot(Vec2,Vec2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle


def knee_to_bar_dist(weightedAvg):
        """Accepts the 8x3 weighted average landmark array. Returns the approximate distance 
        between the knees and the barbell (approximated by hands), taken along the axis normal to the barbell."""

        kneeAvg = weightedAvg[1, 0]
        barAvg = weightedAvg[6, 0]
        kneeToBarDist = distance(kneeAvg,barAvg)
        return kneeToBarDist

def knee_to_shoulder_dist(landmarks):
        """Accepts 33x4 numpy array of landmark coordinates, returns the approximate distance 
        between the knees and the shoulders. 
        """
        leftKnee = landmarks[25, 0:2]
        rightKnee = landmarks[26, 0:2]
        leftShoulder = landmarks[11, 0:2]
        rightShoulder = landmarks[12, 0:2]
        kneeToShoulderDist = max(distance(leftKnee,leftShoulder),distance(rightKnee,rightShoulder))

        return kneeToShoulderDist

def feet_position(self, landmarks):
        """Accepts 33x4 numpy array of landmark coordinates; ensures that feet are roughly 
        shoulder width apart"""
        
        leftFoot = landmarks[27, 0:2]
        rightFoot = landmarks[28, 0:2]
        leftShoulder = landmarks[11, 0:2]
        rightShoulder = landmarks[12, 0:2]
        

        distBetweenShoulders = distance(leftShoulder,rightShoulder)
        distBetweenFeet = distance(leftFoot,rightFoot)

        if (distBetweenShoulders*0.98 <= distBetweenFeet) and (distBetweenFeet <= distBetweenShoulders*1.02):
            return True

        return False

    # def barbell_position(self):
    #     """Takes left- and right-sided landmark coordinates as input, returns the position of the barbell
    #     by approximating the barbell as the straight line which connects both of the hands"""

    # def hips_at_ground(self):
    #     """ Proper deadlift form requires that the hips are below the shoulders and above the knees at the
    #     ground (AKA rest) position. This method checks the relative positions of the three joints ,
    #     returning a pass/fail."""


def neck_angle(self, weightedAvg):
        """Measures the angle made by the neck and the spine. The angle should be approximately equal to zero
        throughout the course of the lift. Returns Pass/Fail. Extract the 2D (x,y) pair which corresponds to each
        of the hip, shoulder, and ear joints. From these pairs, construct vectors which correspond to the spine
        (here, the spine is approximated as the line connecting the hips and shoulders) and the neck
        (similarly corresponding to the line which connects the shoulders to the ears). With these two lines,
        we use the relationship between the dot product of two vectors and the angle of the cosine between them."""

        spineVector = weightedAvg[3, 0:2] - weightedAvg[2,0:2]
        neckVector = weightedAvg[4, 0:2] - weightedAvg[3,0:2]
        degree = angle(spineVector,neckVector)

        if abs(degree) >= 5:
            return False

        return True


"""
Revisions:
    
    This file should be a collection of individual methods. No classes.
    Furthermore, decompose pre-existing methods. I.e., Decompose neck_angles
    into two methods: one which calculates the angle between two vectors, one
    which uses the aforementioned method to specifically return the neck angle.

"""


                

          
           

                    
                
            












            
            
            
            

        
    
    
 
        
 
        
        
        
        
        
    
   
