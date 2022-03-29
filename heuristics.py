"""
Author: Ryan Markson
Email: ryanmarkson1@gmail.com
Created: 3/18/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""
import lm_extractor as lme
from renderer import Renderer
import numpy as np  
from numpy.linalg import norm 
import cv2
from vid_man import VideoManager 




class performance_tracker:
    
    def __init__(self):
    
 

    def knee_to_bar_dist(self, weightedAvg):
        """Accepts the current coordinates of both knees and both hands. Returns the approximate distance 
        between the knees and the barbell (approximated by hands), taken along the axis normal to the barbell."""
        
        kneeAvg = weightedAvg[1,0]
        barAvg = weightedAvg[6,0]
        kneeToBarDist = barAvg - kneeAvg
        # leftKnee = landmarks[25,0:2]
        # rightKnee = landmarks[26,0:2]
        # leftHand = landmarks[15,0:2]
        # rightHand = landmarks[16,0:2]
        # kneeToBarDist = max(norm(leftKnee - leftHand,2),norm(rightKnee-rightHand,2))
        
        
        return kneeToBarDist
        
    def knee_to_shoulder_dist(self, landmarks):
        """Accepts 33x4 numpy array of landmark coordinates, returns the approximate distance 
        between the knees and the shoulders. 
        """
        leftKnee = landmarks[25,0:2]
        rightKnee = landmarks[26,0:2]
        leftShoulder = landmarks[11,0:2]
        rightShoulder = landmarks[12,0:2]
        kneeToShoulderDist = max(norm(leftKnee-leftShoulder,2),norm(rightKnee-rightShoulder,2))
        
        return kneeToShoulderDist
        
        
    def feet_position(self, landmarks):
        """Accepts 33x4 numpy array of landmark coordinates; ensures that feet are roughly 
        shoulder width apart"""
        
        distBetweenShoulders = abs(left[3,1] - right[3,1])
        distBetweenFeet = abs(left[0,1] - right[0,1])
        
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
        
        spineVector = weightedAvg[3,0:2] - weightedAvg[2,0:2]
        neckVector = weightedAvg[4,0:2] - weightedAvg[3,0:2]
        cosine_angle = np.dot(spineVector,neckVector)/(np.dot(spineVector,spineVector)*np.dot(neckVector,neckVector))
        degreee = math.degrees(math.acos(cosine_angle))
        
        if abs(degree) >=5:
            return False
        
        return True
        

    
    def critique_side_view(self, weightedAvg):
        
        #Takes 8x3 numpy array of weighted averages as input
        
 

                        
                if self.knee_to_bar_dist(weightedAvg):  
                        #annotate the image accordingly
               
                if self.neck_angle(weightedAvg):
                        #annotate the image accordingly
                
                

    def critique_front_view(self, landmarks):
        
        #Takes 33x4 numpy array of landmark coordinates as input
        

                if self.feet_position(landmarks):
                        #annotate the image accordingly
                        
     

                

          
           

                    
                
            












            
            
            
            

        
    
    
 
        
 
        
        
        
        
        
    
   