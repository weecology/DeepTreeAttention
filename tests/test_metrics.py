#Test metrics
from DeepTreeAttention.utils import metrics
import pytest

def test_site_confusion():
    y_true = [1,1,2,2]
    y_pred = [0, 1, 1, 2]
    site_list = {0: [0], 1: [1], 2:[1]}
    
    #There are two errors, one error comes from within site
    within_site  = metrics.site_confusion(y_true, y_pred, site_list)
    assert within_site == 0.5
    
    