#MNSIT simulation
from distributed import wait
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from src.data import read_config
from src import start_cluster
import torch
from torch.utils.data import Dataset

class mnist_dataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, index):
        pass
                
class simulation_data(LightningDataModule):
    """A simulation data module"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def download_mnist(self):
        pass
    
    def sample_mnist(self):
        pass
    
    def add_noise(self):
        pass
    
    def swap_labels(self):
        pass
    
    def corrupt(self, df):
        df = self.add_noise(df)
        df = self.swap_labels(df)
    
    def split_train_test(self):
        pass
    
    def setup(self):
        self.mnist_data = self.download_mnist()
        self.true_state = self.sample_mnist(self.mnist_data)
        self.corrupted_mnist = self.corrupt(self.true_state)
        self.train, self.test = self.split_train_test(self.corrupted_mnist)
        
        self.train_ds = mnist_dataset(self.train)
        self.val_ds = mnist_dataset(self.test)
        
    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
class simulator():
    def __init__(self, model, log = True):
        """Simulation object
        Args:
            model: a pytorch lightning model to run
            comet_experiment: an optional comet logger 
        """
        self.config = read_config("simulation.yml")
        self.model = model
        if log:
            self.comet_experiment = CometLogger(project_name="DeepTreeAttention", workspace=self.config["comet_workspace"],auto_output_logging = "simple")
            self.comet_experiment.experiment.add_tag("simulation")
        
    def generate_data(self):
        self.data_module = simulation_data(config=self.config)
        
    def train(self):
        #Create trainer
        trainer = Trainer(
            gpus=self.config["gpus"],
            fast_dev_run=self.config["fast_dev_run"],
            max_epochs=self.config["epochs"],
            accelerator=self.config["accelerator"],
            checkpoint_callback=False,
            logger=self.comet_experiment)
        
        trainer.fit(self.model, datamodule=self.data_module)
    
    def outlier_detection(df, true_state):
        """How many of the noise data are currently labeled as outliers"""
        pass
    
    def label_switching(df, true_state):
        pass
    
    def evaluate(self):
        df = self.model.predict(self.val_ds)
        outlier_results = self.outlier_detection(df, self.data_module.true_state)
        label_switching = self.label_switching(df, self.data_module.true_state)
        
        results = pd.concat(list(outlier_results, label_switching))
        return results
        
    
def run(ID):
    sim = simulator()
    sim.generate_data()
    sim.train()
    results = sim.evaluate()
    results["simulation_id"] = ID
    return results
    
if __name__ == "__main__":
    client = start_cluster.start(gpus=2)
    futures = []
    for x in range(10):
        future = client.submit(run, ID=x)
        futures.append(future)
    wait(futures)
    resultdf = []
    for x in futures:
        result = x.result()
        resultdf.append(result)
    
    resultdf = pd.concat(resultdf)
    resultdf.to_csv("data/processed/simulation.csv")
    