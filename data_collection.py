import cv2
import numpy as np
import os
import time
import mediapipe as mp

# Path for exported data, numpy arrays
path = "/home/ubuntu/Documents/Sign-Language Project/Dataset"
DATA_PATH_NUMBERS = os.path.join(path, "numbers")
DATA_PATH_ALPHA = os.path.join(path, "alphabets")

# Actions that we try to detect
actions_num = np.array(['zero', 'one', 'two','three','four', 'five', 'six', 'seven', 'eight', 'nine']) # For nums
actions_alpha = np.array(['space', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                          'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
                         ])   # For alphabets

# 40 videos
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 5

# Folder start
## Change this variable according to your starting folder
'''
Soham: 0 - 39
Pritish: 40 - 79
Sharvin: 80 - 119
Shantanu: 120 - 159
'''
start_folder = 220


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def create_folders():
    for action in actions_num: 
        #dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        for sequence in range(start_folder, start_folder+no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH_NUMBERS, action, str(sequence)))
            except:
                pass
        

    for action in actions_alpha: 
        #dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        for sequence in range(start_folder, start_folder+no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH_ALPHA, action, str(sequence)))
            except:
                pass


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
#                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                              ) 
    # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                              ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# def plotter(results):
#     try:
#         data = results
#         plt.figure()
#         ax = plt.axes(projection='3d')
#         x = data[:, 0]
#         y = data[:, 1]
#         z = data[:, 2]
#         ax.plot3D(x, y, z, 'red')
#         ax.set_zlabel('Z-Axis')
#         ax.set_ylabel('Y-Axis')
#         ax.set_xlabel('X-Axis')
#         ax.view_init(240, -90)
#         plt.draw()
#         plt.pause(.01)
#         # plt.plot(data,)
#         plt.show()
#     except:
#         pass

def test_cv2():

    cam = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            #width  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)   # float width
            #height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float height
            #print(width)
            #print(height)
            #break
            # Read feed
            ret, frame = cam.read()
            #print(frame)
            #break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            #print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # keypoints = extract_keypoints(results)
            # all_zeros = not np.any(keypoints)
            # if not all_zeros:
            #     plotter(keypoints)

            # Show to screen
            cv2.imshow('Sign Detector', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()


    pose = []
    for res in results.pose_landmarks.landmark:
        test = np.array([res.x, res.y, res.z, res.visibility])
        pose.append(test)



    rest = extract_keypoints(results)

    np.save('0',rest)


# Backup function incase of some mistake in a particular folder.
def backup_folders():

    global start_folder
    global no_sequences

    folder = input("Enter the symbol name: ")
    for sequence in range(start_folder, start_folder + no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH_ALPHA, folder, str(sequence)))
        except:
            pass



def create_nums():
    # For Numbers
    global start_folder
    global no_sequences
    global mp_drawing
    global mp_holistic
    global sequence_length
    global actions_num
    global DATA_PATH_NUMBERS
    global path
    flag = False
    cap = cv2.VideoCapture(0)

    print("It's {} time baby".format(actions_num[9]))

    j = 0
    while j<20:
        j+=1

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder + no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (400, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image,'Collecting frames for {} Video Number {}'.format(actions_num[9], sequence), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    #cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions_num[9], sequence), (15,12), 
                    #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('Sign_Detector', image)
                    cv2.waitKey(3000)
                else: 
                    cv2.putText(image, str(frame_num), (450, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions_num[9], sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('Sign_Detector', image)

                # NEW Export keypoints
                #keypoints = extract_keypoints(results)
                #npy_path = os.path.join(DATA_PATH_NUMBERS, actions_num[9], str(sequence), str(frame_num))
                #np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    flag = True
                    break

            if flag:
                break
            
        cap.release()
        cv2.destroyAllWindows()

# For alphabets
def create_alphabets():

    global start_folder
    global no_sequences
    global mp_drawing
    global mp_holistic
    global sequence_length
    global actions_alpha
    global DATA_PATH_ALPHA
    global path
    flag = False
    cap = cv2.VideoCapture(0)

    print("It's {} time baby".format(actions_alpha[23]))
    j = 0
    while j<20:
        j+=1
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions_alpha[23], sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('Sign_Detector', image)
                    cv2.waitKey(3000)
                else: 
                    cv2.putText(image, str(frame_num), (120,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions_alpha[23], sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('Sign_Detector', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH_ALPHA, actions_alpha[23], str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    flag = True
                    break

            if flag:
                break

        cap.release()
        cv2.destroyAllWindows()

        
option = input("1) Test your Camera using cv2\n2) Create dataset for Numbers\n3) Create dataset for Alphabets\n4) Create folders\n5) Backup folder\nEnter your choice: ")

if option == '1':
    test_cv2()
elif option == '2':
    create_nums()
elif option == '3':
    # print(actions_alpha[13])
    create_alphabets()
elif option == '4':
    create_folders()
else :
    backup_folders()


# label_map = {label:num for num, label in enumerate(actions_num)}


#     sequences, labels = [], []


#     for action in actions_num:
#     #     for numbers
#         for sequence in np.array(os.listdir(os.path.join(DATA_PATH_NUMBERS, action))).astype(int):
#             window = []
#             for frame_num in range(sequence_length):
#                 res = np.load(os.path.join(DATA_PATH_NUMBERS, action, str(sequence), "{}.npy".format(frame_num)))
#                 window.append(res)
#             sequences.append(window)
#             labels.append(label_map[action])
        
            
#     np.save('sign_gest',np.array(sequences))
#     np.save('labels',np.array(labels))



# space a b c d e f g h i l m n o r s t u v w x y 

# Remaining: j k-done p-done q z
# Below ones are not working correctly
# j,k,p,q,z
