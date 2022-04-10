import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from mediapipe_functions import Mediapipe_functions
from textblob import TextBlob
import keyboard

root = tk.Tk()
root.geometry("1300x800")
root.title("Communication for the unspoken")
root.resizable(False, False)
media = Mediapipe_functions()
file = "database.json"
new_word = ""
new_sent = ""
text = tk.StringVar()
wind_close = True

def textBlob_spellcheck(word):
    textBlb = TextBlob(word)
    textCorrected = textBlb.correct()
    return textCorrected

def make_sentence(sign_map):
    global new_word
    global new_sent
    global text
    letter = max(sign_map, key=sign_map.get)
    print("debug: ",letter)

    if letter == 'space' and new_sent.substr(len(new_sent)-1) == " ":
	pass
    elif new_word.substr(len(new_sent)-1) == letter:
        if new_word.substr(len(new_sent)-2, len(new_sent)-1) != letter:
            new_word += letter
    elif letter == 'space':
        new_word = textBlob_spellcheck(new_word)
        if new_sent == "":
            new_sent = str(new_word)
        else:
            new_sent = new_sent + " " + str(new_word)
        print(new_word, end=" ")
        new_word = ''
    else:
        new_word += letter
    return new_sent

def test_predictions_2():
    count = 0
    mapdict = {}
    sequence = []
    threshold = 0.8
    tex = ""
    cap = cv2.VideoCapture(0)
    prev_pred = " "
    prev_thresh = 0.0
    # Set mediapipe model
    with media.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame1 = cap.read()
            # Make detections
            image1, results = media.mediapipe_detection(frame1, holistic)
            # Draw landmarks
            # draw_styled_landmarks(image, results)
            # prediction logic
            keypoints = media.extract_keypoints(results)
            all_zeros = not np.any(keypoints)
            if not all_zeros:
                sequence.append(keypoints)
                sequence = sequence[-5:]
                if len(sequence) == 5:
                    count += 1
                    res = media.model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        pred = media.actions[np.argmax(res)]
                        # current_thresh = res[np.argmax(res)]
                        # if prev_pred != pred or (prev_pred == pred and (current_thresh >= prev_thresh-0.05 and current_thresh<=prev_thresh+0.05)):
                        cv2.putText(image1, pred, (400, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3, cv2.LINE_AA)
                        if pred in mapdict:
                            temp = mapdict[pred]
                            mapdict[pred] = temp + 1
                        else:
                            mapdict[pred] = 1

                        if count >= 10:
                            count = 0
                            tex = make_sentence(mapdict)
                            mapdict = {}
                        # prev_pred = pred
                        # prev_thresh = current_thresh
            cv2.putText(image1,tex, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # Show to screen
            cv2.imshow('Sign_Detector', image1)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def gestures_to_script():
    global start_button
    global sign_text
    global enter_button

    def hide_widgets(widg1, widg2):
        widg1.place_forget()
        widg2.place_forget()
        print("Done")

    hide_widgets(sign_text, enter_button)

    # Workspace
    label = tk.Label(frame, text="Gestures to script", font="courier 15 bold", justify="center")
    label.place(relx=0.225, rely=0.0)
    # text_label.place(relx=0.01, rely=0.8, relwidth=0.6, relheight=0.1)
    # Start button
    start_button.config(command=test_predictions_2)
    start_button.place(relx=0.14, rely=0.73, relwidth=0.15, relheight=0.05)
    # Stop button
    # stop_button.config(command=close_cam)
    # stop_button.place(relx=0.31, rely=0.73, relwidth=0.15, relheight=0.05)

def connectpoints(res_list):
    global wind_close
    f = open(file, 'r')
    data_file = json.load(f)
    fig = plt.figure(figsize=(5, 5), dpi=100)
    # figManager = plt.get_current_fig_manager()
    # backend = plt.get_backend()
    # print(backend)
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    # figManager.window.showMaximized()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    wind_close = True
    for symbol in res_list:
        if (symbol == " "):
            symbol = "space"

        for i in range(5):
            results = data_file[symbol][str(i)]
            ax = fig.add_subplot(111, projection="3d")
            plt.title(symbol)
            # Thumb with base point
            x0, x1, x2, x3, x4 = results[0][0], results[1][0], results[2][0], results[3][0], results[4][0]
            y0, y1, y2, y3, y4 = results[0][1], results[1][1], results[2][1], results[3][1], results[4][1]
            z0, z1, z2, z3, z4 = results[0][2], results[1][2], results[2][2], results[3][2], results[4][2]

            # Index Finger
            x5, x6, x7, x8 = results[5][0], results[6][0], results[7][0], results[8][0]
            y5, y6, y7, y8 = results[5][1], results[6][1], results[7][1], results[8][1]
            z5, z6, z7, z8 = results[5][2], results[6][2], results[7][2], results[8][2]

            # Middle Finger
            x9, x10, x11, x12 = results[9][0], results[10][0], results[11][0], results[12][0]
            y9, y10, y11, y12 = results[9][1], results[10][1], results[11][1], results[12][1]
            z9, z10, z11, z12 = results[9][2], results[10][2], results[11][2], results[12][2]

            # Ring Finger
            x13, x14, x15, x16 = results[13][0], results[14][0], results[15][0], results[16][0]
            y13, y14, y15, y16 = results[13][1], results[14][1], results[15][1], results[16][1]
            z13, z14, z15, z16 = results[13][2], results[14][2], results[15][2], results[16][2]

            # pinky finger with base point
            x17, x18, x19, x20 = results[17][0], results[18][0], results[19][0], results[20][0]
            y17, y18, y19, y20 = results[17][1], results[18][1], results[19][1], results[20][1]
            z17, z18, z19, z20 = results[17][2], results[18][2], results[19][2], results[20][2]

            # Joining finger base points
            ax.plot3D([x2, x5, x9, x13, x17], [y2, y5, y9, y13, y17], [z2, z5, z9, z13, z17], 'red', linewidth=2)

            # Joining finger base points to base of palm
            ax.plot3D([x0, x5], [y0, y5], [z0, z5], 'red', linewidth=1)
            ax.plot3D([x0, x9], [y0, y9], [z0, z9], 'red', linewidth=1)
            ax.plot3D([x0, x13], [y0, y13], [z0, z13], 'red', linewidth=1)

            # PLotting each finger
            ax.plot3D([x0, x1, x2, x3, x4], [y0, y1, y2, y3, y4], [z0, z1, z2, z3, z4], 'k-', linewidth=4)
            ax.plot3D([x5, x6, x7, x8], [y5, y6, y7, y8], [z5, z6, z7, z8], 'k-', linewidth=4)
            ax.plot3D([x9, x10, x11, x12], [y9, y10, y11, y12], [z9, z10, z11, z12], 'k-', linewidth=4)
            ax.plot3D([x13, x14, x15, x16], [y13, y14, y15, y16], [z13, z14, z15, z16], 'k-', linewidth=4)
            ax.plot3D([x0, x17, x18, x19, x20], [y0, y17, y18, y19, y20], [z0, z17, z18, z19, z20], 'k-', linewidth=4)
            ax.disable_mouse_rotation()
            ax.view_init(240, -90)
            plt.axis('off')
            if wind_close:
                plt.pause(0.2)
                # mng.frame.Maximize(True)
                # mng.close_screen_toggle()
                # mng.partial_screen_toggle()
                # figManager.window.showMaximized()
                # mng.resize(*mng.window.)
                # mng.resize(*mng.window.minimized())
                # keyboard.send("windows+down")
                wind_close = False
            else:
                plt.pause(0.2)
            canvas.draw()
            plt.clf()
            canvas.get_tk_widget().place(relx=0.1, rely=0.1)
    f.close()
    plt.close()
    canvas.get_tk_widget().destroy()

def script_to_gestures():
    global start_button
    global stop_button
    global text_label
    start_button.place_forget()
    # def hide_buttons(widg1,widg2,widg3):
    #     widg1.place_forget()
    #     widg2.place_forget()
    #     widg3.place_forget()
    #     #print("done")
    # hide_buttons(start_button, stop_button, text_label)

    label = tk.Label(frame, text="Script to gestures", font="courier 15 bold", justify="center")
    label.place(relx=0.225, rely=0.0)
    sign_text.place(relx=0.01, rely=0.8, relwidth=0.6, relheight=0.1)
    enter_button.config(command=lambda: connectpoints(list(sign_text.get("1.0", "end-1c"))))
    enter_button.place(relx=0.555, rely=0.8, relwidth=0.1, relheight=0.1)

#elements
label = ttk.Label(root, text="Communication for\nthe unspoken", font="courier 15 bold", justify="center")
button1 = ttk.Button(root, text="Gestures to script", command=test_predictions_2)
button2 = ttk.Button(root, text="Script to gesture", command=script_to_gestures)
button3 = ttk.Button(root, text="Quit", command=quit)
partition = ttk.Separator(root, orient='vertical')

image = Image.open("HeartHackers.jpeg")
resize_image = image.resize((70, 70))
img = ImageTk.PhotoImage(resize_image)
photo = tk.Label(image=img)
label_copyright = ttk.Label(root, text="© 2021 HeartHackers India,\n      Inc. All rights reserved.")
frame = ttk.Frame(root)

# Gesture to script
start_button = ttk.Button(frame, text="Start camera")
# stop_button = ttk.Button(frame, text="Stop camera")
# text_label=tk.Label(frame , textvariable=text, font="arial 18 bold", justify="center", height = 5, width = 25, bg = "white")

# Script tp gesture
sign_text = tk.Text(frame, height=4, width=25, bg="white")
enter_button = tk.Button(frame, text="Enter")

#placement
label.place(relx=0.02 ,rely=0.02)
button1.place(relx=0.02, rely=0.1, relwidth=0.15, relheight=0.1)
button2.place(relx=0.02, rely=0.22, relwidth=0.15, relheight=0.1)
button3.place(relx=0.02, rely=0.34, relwidth=0.15, relheight=0.1)
partition.place(relx=0.2, rely=0, relwidth=1, relheight=1)
frame.place(relx=0.3,rely=0.05, relheight=1,relwidth=1)
photo.place(relx=0.014, rely=0.9)
label_copyright.place(relx=0.069, rely=0.93)
root.mainloop()
