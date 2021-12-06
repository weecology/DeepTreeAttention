#Test CHM height rules
import os
import pandas as pd
from src import CHM, data

def test_height_rules():   
    ROOT = os.path.dirname(os.path.dirname(CHM.__file__))
    config = data.read_config("{}/config.yml".format(ROOT))
    df = pd.DataFrame({"CHM_height":[11,20,5, 0.5, 10],"height":[6, 19, 7, 5, None]})
    df = CHM.height_rules(df, min_CHM_height=1, max_CHM_diff=4, CHM_height_limit=8)
    assert df.shape[0] == 3
    