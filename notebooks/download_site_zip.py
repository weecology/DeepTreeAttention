import subprocess
import os
from src.model_list import species_model_paths

for site in ["CLBJ","BART","UKFS","BLAN","MLBS"]:
    # Download model
    p = subprocess.Popen(
        "scp hpg:{} /Users/benweinstein/Dropbox/Weecology/Species/SpeciesMaps/snapshots/".format(species_model_paths[site]), shell=True)
    p.wait()
    # Zip predictions
    basename = os.path.splitext(os.path.basename(species_model_paths[site]))[0]
    zipfilename = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.zip".format(
        site, basename, site)
    if not os.path.exists("/Users/benweinstein/Dropbox/Weecology/Species/SpeciesMaps/{}.zip".format(site)):
        # Copy zip locally
        print(site)
        p = subprocess.Popen(
            "scp hpg:{} /Users/benweinstein/Dropbox/Weecology/Species/SpeciesMaps/".format(zipfilename), shell=True)
        p.wait()
