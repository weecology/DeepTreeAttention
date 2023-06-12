#Check for invalid tar files
import glob
import os
import tempfile
import tarfile
import traceback
from src import start_cluster
from distributed import wait, as_completed

def check_tar(tarfilename):
    tmpdir = tempfile.TemporaryDirectory()
    try:
        with tarfile.open(tarfilename, 'r') as archive:
            archive.extractall(tmpdir.name)  
    except tarfile.ReadError:
        os.remove(tarfilename)
        tmpdir.cleanup()
        return tarfilename
    return None


#client = start_cluster.start(cpus=50, mem_size="4GB")
shapefiles = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/*/shp/*.shp")

futures = []
for shapefile in shapefiles:
    #get tarfile from shp
    tarfilename = shapefile.replace("shp","tar")
    tarfilename = "{}.tar.gz".format(os.path.splitext(tarfilename)[0])    
    if not os.path.exists(tarfilename):
        print("{} is missing".format(tarfilename))
        os.remove(shapefile)
        continue
    #future = client.submit(check_tar, tarfilename)
    #futures.append(future)

#for x in as_completed(futures):
    #try:
        #x.result()
    #except:
        #traceback.print_exc()
