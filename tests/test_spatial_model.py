#tests models/spatial.py
import numpy as np
from src.models import spatial
from pytorch_lightning import Trainer

def test_fit():
    train_sensor_score = np.random.multinomial(1,[1/6,2/6,3/6], (10)).astype(np.float32)
    train_neighbor_score = np.random.multinomial(1,[1/6,2/6,3/6], (10)).astype(np.float32)
    val_sensor_score = np.random.multinomial(1,[1/6,2/6,3/6], (10)).astype(np.float32)
    val_neighbor_score = np.random.multinomial(1,[1/6,2/6,3/6], (10)).astype(np.float32)
    train_labels = np.random.randint(low=0, high=3, size = 10)
    val_labels = np.random.randint(low=0, high=3, size = 10)
    
    m = spatial.spatial_fusion(train_sensor_score, train_neighbor_score, val_sensor_score, val_neighbor_score, train_labels, val_labels)
    trainer = Trainer(fast_dev_run=True, checkpoint_callback=False)
    trainer.fit(m)