#Test year model
from src.models import year
from pytorch_lightning import Trainer
def test_year_model(dm, config):
    model = year.year_converter(csv_file=dm.train_file, config=config, band = 1, year = 2018)
    trainer = Trainer()
    trainer.fit(model)