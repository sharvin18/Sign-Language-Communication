import cv2
import numpy as np
import os
import time
import mediapipe as mp
# from nlp_testing import hello
from mediapipe_functions import mediapipe_functions
import json
from os import path

sequence_length = 5
# mp_holistic = mp.solutions.holistic # Holistic model
# mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# Actions that we try to detect
actions_num = np.array(['0', '1', '2','3','4', '5', '6', '7', '8', '9']) # For nums
actions_alpha = np.array(['space', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                          'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
                         ])   # For alphabets
def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(21*3)
    return rh

file = "database1.json"
def create_hand_database():
    
    global sequence_length
    global actions_num
    flag = False
    cap = cv2.VideoCapture(0)
    mdp = mediapipe_functions()
    
    with mdp.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through sequences aka videos
        for sequence in range(5):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mdp.mediapipe_detection(frame, holistic)

                # Draw landmarks
                mdp.draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions_num[0], sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('Sign_Detector', image)
                    cv2.waitKey(3000)
                else: 
                    cv2.putText(image, str(frame_num), (120,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions_num[0], sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('Sign_Detector', image)

                if sequence == 2:

                    if frame_num == 0:
                        keypoints0 = extract_keypoints(results)
                    elif frame_num == 1:
                        keypoints1 = extract_keypoints(results)
                    elif frame_num == 2:
                        keypoints2 = extract_keypoints(results)
                    elif frame_num == 3:
                        keypoints3 = extract_keypoints(results)
                    elif frame_num == 4:
                        keypoints4 = extract_keypoints(results)
                   
                        data = {
                            "0" : keypoints0.tolist(),
                            "1" : keypoints1.tolist(),
                            "2" : keypoints2.tolist(),
                            "3" : keypoints3.tolist(),
                            "4" : keypoints4.tolist()  
                        }
                        
                        if path.isfile(file) is False:
        
                            dictionary = {}
                            dictionary[actions_num[0]] = data
                            json_object = json.dumps(dictionary, indent = 4)

                        else:
                            with open(file, 'r') as openfile:
                                data_file = json.load(openfile)
                                # print(data_file["user"]["0"])
                                data_file[actions_num[0]] = data
                                json_object = json.dumps(data_file, indent = 4)
                                openfile.close()
                                

                        with open(file, "w") as outfile:
                                outfile.write(json_object)
                                outfile.close()

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    flag = True
                    break

            if flag:
                break
            
        cap.release()
        # cv2.destroyAllWindows()


inp = input("Hit enter when you want to start")
create_hand_database()