import os
import glob
import shutil
def delete_folders(directory):
    for folder in os.listdir(directory):
        if folder.startswith('20') and os.path.isdir(os.path.join(directory, folder)):
            folder_path = os.path.join(directory, folder)
            print(folder_path)
            #shutil.rmtree(folder_path) 
            try:
                os.rmdir(folder_path)
            except:
                pass



dirs = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/*/")
for dir in dirs:
    for subdir in os.walk(dir):
        delete_folders(dir)