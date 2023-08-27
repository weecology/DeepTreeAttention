import tarfile
import glob
import tempfile
import os

tars = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/TEAK/tar/*")

for tar in tars:
    print(tar)
    try:
        with tarfile.open(tar, 'r') as archive:
            with tempfile.TemporaryDirectory() as temp_dir:
                archive.extractall(temp_dir)
    except tarfile.ReadError:
        basename = os.path.splitext(os.path.basename(tar))[0].split(".")[0]
        os.remove(tar)
        shp_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/TEAK/shp/{}.shp".format(basename)
        os.remove(shp_path)
        print("deleteing {}".format(tar))
        print("deleteing {}".format(shp_path))

        