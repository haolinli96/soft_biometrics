import os
import cv2

file_dir = "gender_data"
count = 0
for root, dirs, files in os.walk(file_dir):
    for dir_name in dirs:
        dir_path = os.path.join(file_dir, dir_name)
        for root_1, dirs_1, files_1 in os.walk(dir_path):
            for file_name in files_1:
                file_path = os.path.join(dir_path, file_name)
                #print(file_path)
                img = cv2.imread(file_path)
                if img is None:
                    continue
                u, v, d = img.shape
                #count += 1
                print(u, v)
                resize = cv2.resize(img, (224, 224))

                cv2.imwrite(file_path, resize)

#print(count)