#MNSIT simulation
from distributed import wait
import os
import numpy as np
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer
from src.data import read_config
from src import start_cluster
from src.models.simulation import autoencoder
import torch
import torchvision
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(__file__))

class mnist_dataset(Dataset):
    """Yield an MNIST instance"""
    def __init__(self, df):
        self.annotations = df
    
    def __len__(self):
        return self.annotations.shape[0]
        
    def __getitem__(self, index):
        image_path = self.annotations.image_path.iloc[index]    
        image = io.imread(image_path)
        observed_label = self.annotations.observed_label.iloc[index]      
        true_label = self.annotations.true_label.iloc[index]      
        
        image = torch.tensor(image).unsqueeze(0).float()
        observed_label = torch.tensor(observed_label)
        true_label = torch.tensor(true_label)
        
        return image, observed_label, true_label        
               
    
class simulation_data(LightningDataModule):
    """A simulation data module"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def download_mnist(self):
        self.raw_ds = torchvision.datasets.MNIST('{}/data/simulation/'.format(ROOT), train=True, download=True)                            
        
        #Grab 500 examples and write jpegs and class labels
        class_labels = {}
        for x in range(10):
            class_labels[x] = 0
        
        image_paths = []
        labels = []
        for x in range(len(self.raw_ds)):
            image, label = self.raw_ds[x]
            if class_labels[label] < 500:
                class_labels[label] = class_labels[label] + 1
                labels.append(label)
                fname = "{}/data/simulation/{}_{}.png".format(ROOT, label,class_labels[label])
                image_paths.append(fname)
                image.save('{}'.format(fname))
            
        df = pd.DataFrame({"image_path":image_paths, "label":labels})
        
        return df 
    
    def corrupt(self):
        
        self.raw_df["true_label"] = self.raw_df["label"]
        self.raw_df["observed_label"] = self.raw_df["true_label"]
        
        novel_set = self.raw_df[self.raw_df.label.isin([8,9])]
        novel_set = novel_set.groupby("label").apply(lambda x: x.head(self.config["novel_class_examples"]))
        in_set = self.raw_df[~self.raw_df.label.isin([8,9])]
        
        #label swap within in set
        labels_to_corrupt = in_set.groupby("label").apply(lambda x: x.sample(frac=self.config["proportion_switch"]))
        labels_to_corrupt["observed_label"] = labels_to_corrupt.label.apply(lambda x: np.random.choice(range(8)))
        
        uncorrupted_labels = in_set[~in_set.image_path.isin(labels_to_corrupt.image_path)]
        corrupted_data = pd.concat([uncorrupted_labels, labels_to_corrupt, novel_set])
        
        return corrupted_data
    
    def split(self):
        """Split train/test"""
        
        train = self.corrupted_mnist.groupby("observed_label").sample(frac = 0.9)
        test = self.corrupted_mnist[~self.corrupted_mnist.image_path.isin(train.image_path)]
        
        return train, test
    
    def setup(self, stage=None):
        self.raw_df = self.download_mnist()
        self.corrupted_mnist = self.corrupt()   
        self.train, self.test = self.split()
        self.train_ds = mnist_dataset(self.train)
        self.val_ds = mnist_dataset(self.test)
        self.num_classes = len(np.unique(self.train.label))
        
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
    def __init__(self, config, log = True):
        """Simulation object
        Args:
            comet_experiment: an optional comet logger 
            config: path to config.yml
        """
        self.config = read_config(config)
        self.log = log
        if self.log:
            self.comet_experiment = CometLogger(project_name="DeepTreeAttention", workspace=self.config["comet_workspace"],auto_output_logging = "simple")
            self.comet_experiment.experiment.add_tag("simulation")
    
    def generate_data(self):
        self.data_module = simulation_data(config=self.config)
        self.data_module.setup()
    
    def create_model(self):
        self.model = autoencoder(bands=1, classes=self.data_module.num_classes, config=self.config)
        
    def train(self):
        #Create trainer
        self.trainer = Trainer(
            gpus=self.config["gpus"],
            fast_dev_run=self.config["fast_dev_run"],
            max_epochs=self.config["epochs"],
            accelerator=self.config["accelerator"],
            checkpoint_callback=False,
            logger=self.comet_experiment)
        
        self.trainer.fit(self.model, datamodule=self.data_module)
    
    def outlier_detection(self, results):
        """Given a set of predictions, label outliers"""
        threshold = results.loss.quantile(self.config["outlier_threshold"])
        print("Reconstruction threshold is {}".format(threshold))
        results["outlier"] = results[results.loss > threshold]
        
        return results
    
    def label_switching(self):
        """Detect clusters and identify mislabeled data"""
        pass
    
    
    def evaluate(self):
        observed_y = []
        yhat = []
        y = []
        autoencoder_loss = []
        self.model.eval()
        for batch in self.model.val_dataloader():
            images, observed_labels, true_labels = batch
            observed_y.append(observed_labels)
            y.append(true_labels)
            #trigger activation hook
            if next(self.model.parameters()).is_cuda:
                images = images.cuda()
            with torch.no_grad():
                for image in images:
                    image_yhat, classification_yhat = self.model(image.unsqueeze(0))
                    yhat.append(classification_yhat)
                    loss = F.mse_loss(image_yhat, image)    
                    autoencoder_loss.append(loss.numpy())
           
        yhat = np.concatenate(yhat)
        yhat = np.argmax(yhat, 1)
        autoencoder_loss = np.asarray(autoencoder_loss)
        
        results = pd.DataFrame({"true_label":y,"observed_label": observed_y,"predicted_label":yhat, "autoencoder_loss": autoencoder_loss})
        results = self.outlier_detection(results)
        
        if self.log:
            self.comet_experiment.experiment.log_table("results.csv",results)
        
        #Mean Proportion of true classes are correct
        mean_accuracy = results.filter(~results.true_label.isin([8,9])).groupby("label").apply(lambda x: x.true_label == x.predicted_label).mean()
        if self.log:
            self.comet_experiment.log_metrics("Mean accuracy", mean_accuracy)
        
        true_outliers = results[~results.true_label == results.observed_label]
        #inset data does not have class 8 ir 9
        inset = true_outliers[~true_outliers.true_label.isin(8,9)]
        outlier_accuracy = sum(inset.outlier)/inset.shape[0]
        outlier_precision = sum(inset.outlier)/results.filter(~results.true_label.isin([8,9])).shape[0]
        
        if self.log:
            self.comet_experiment.experiment.log_metric("outlier_accuracy", outlier_accuracy)
            self.comet_experiment.experiment.log_metric("outlier_precision", outlier_precision)
            
        #TODO 
        #label_switching = self.label_switching(df, self.data_module.true_state)
        
        return pd.DataFrame({"outlier_accuracy": outlier_accuracy, "outlier_precision": outlier_precision, "classification_accuracy": mean_accuracy})
        
def run(ID, config_path):
    sim = simulator(config_path)    
    sim.generate_data()
    sim.create_model()
    sim.train()
    results = sim.evaluate()
    results["simulation_id"] = ID
    return results
    
if __name__ == "__main__":
    client = start_cluster.start(gpus=2)
    futures = []
    for x in range(10):
        future = client.submit(run, ID=x, config_path="simulation.yml")
        futures.append(future)
    wait(futures)
    resultdf = []
    for x in futures:
        result = x.result()
        resultdf.append(result)
    
    resultdf = pd.concat(resultdf)
    resultdf.to_csv("data/processed/simulation.csv")
    