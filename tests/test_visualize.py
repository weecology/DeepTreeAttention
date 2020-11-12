#test visualize
from DeepTreeAttention.visualization import visualize

def test_error_crown_position():
    y_true = [0,1,2,2,2]
    y_pred = [1,2,2,2,2]
    box_index = [0,2,1,3,4]
    canopydict = {0:"b",1:"a", 2:"b",3:"b",4:"a"}
    visualize.error_crown_position(y_true, y_pred, box_index, canopydict)