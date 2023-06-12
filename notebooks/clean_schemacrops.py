import glob
import os
import shutil

folders = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/*")
for folder in folders:
        #try:
                #os.mkdir(os.path.join(folder,"tar"))
                #os.mkdir(os.path.join(folder,"shp"))
        #except:
                #pass
        #tars = glob.glob(os.path.join(folder,"*.tar*"))
        #print("There are {} tars in folder {}".format(len(tars),folder))
        #for tar in tars:
                #a=shutil.move(tar, os.path.join(folder,"tar",os.path.basename(tar)))
                
        #shps = glob.glob(os.path.join(folder,"*.shp"))
        #print("There are {} shps in folder {}".format(len(shps),folder))
        
        #for shp in shps:
                #a=shutil.move(shp, os.path.join(folder,"shp",os.path.basename(shp)))       
                
        #shps = glob.glob(os.path.join(folder,"*.prj"))
        #for shp in shps:
                #a=shutil.move(shp, os.path.join(folder,"shp",os.path.basename(shp)))                     
        
        #shps = glob.glob(os.path.join(folder,"*.shx"))
        #for shp in shps:
                #s =shutil.move(shp, os.path.join(folder,"shp",os.path.basename(shp)))               
        
        #shps = glob.glob(os.path.join(folder,"*.dbf"))
        #for shp in shps:
                #a = shutil.move(shp, os.path.join(folder,"shp",os.path.basename(shp)))
        
        #shps = glob.glob(os.path.join(folder,"*.cpg"))
        #for shp in shps:
                #a = shutil.move(shp, os.path.join(folder,"shp",os.path.basename(shp)))  
        
        npys = glob.glob(os.path.join(folder,"*.npy"))
        print("There are {} npys in folder {}".format(len(npys),folder))
        
        a = [os.remove(f) for f in npys]
