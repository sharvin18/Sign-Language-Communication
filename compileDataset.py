import numpy as np
import os


actions = np.array(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine','space', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                          'h', 'i', 'l', 'm', 'n', 'o',
                          'r', 's', 't', 'u', 'v', 'w', 'x', 'y'])  

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
data_path = r'/home/ubuntu/Documents/Sign-Language Project/Dataset-Partial'

sequence_length = 5
for action in actions:
    print(action)
    for sequence in np.array(os.listdir(os.path.join(data_path, action))).astype(int):
      window = []
      for frame_num in range(sequence_length):
        res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame_num)))
        window.append(res)
      sequences.append(window)
      labels.append(label_map[action])

np.save('sequences',np.array(sequences))
np.save('labels',np.array(labels))
print("done")
