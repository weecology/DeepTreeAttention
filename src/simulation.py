#MNSIT simulation
from matplotlib import pyplot as plt
import os
import numpy as np
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from src.models.outlier_detection import autoencoder
from src import outlier
from src import visualize

from PIL import Image
import torch
import torchvision
import pandas as pd

from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms import *
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))

class mnist_dataset(Dataset):
    """Yield an MNIST instance"""
    def __init__(self, df):
        self.annotations = df
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    def __len__(self):
        return self.annotations.shape[0]
        
    def __getitem__(self, index):
        image_path = self.annotations.image_path.iloc[index]    
        image = Image.open(image_path)
        image = np.array(image)
        image = self.transforms(image)
        observed_label = self.annotations.observed_label.iloc[index]      
        label = self.annotations.label.iloc[index]      
        
        #If image_corrupt is true, shuffle pixels
        image_corrupt = self.annotations.image_corrupt.iloc[index]
        if image_corrupt:
            # With view
            idx = torch.randperm(image.nelement())
            image = image.view(-1)[idx].view(image.size())
                    
        observed_label = torch.tensor(observed_label)
        label = torch.tensor(label)
        
        return index, image, observed_label, label        
               
class simulation_data(LightningDataModule):
    """A simulation data module"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tmpdir = tempfile.TemporaryDirectory()
        
    def download_mnist(self):
        
        self.raw_ds = torchvision.datasets.MNIST('{}/data/simulation/'.format(ROOT), train=True, download=True)                            
        
        #Grab examples and write pngs and class labels
        class_labels = {}
        for x in range(10):
            class_labels[x] = 0
        
        image_paths = []
        labels = []
        for x in range(len(self.raw_ds)):
            image, label = self.raw_ds[x]
            if class_labels[label] < self.config["samples"]:
                class_labels[label] = class_labels[label] + 1
                labels.append(label)
                fname = "{}/{}_{}.png".format(self.tmpdir.name, label,class_labels[label])
                image_paths.append(fname)
                image.save(fname)
            
        df = pd.DataFrame({"image_path":image_paths, "label":labels})
        
        return df 
    
    def corrupt_and_split(self):
        """Switch labels of some within sample class and add some novel classes"""
        self.raw_df["observed_label"] = self.raw_df["label"]
        
        #Split clean data into train test
        in_set = self.raw_df[~self.raw_df.label.isin([8,9])]
        in_set["outlier"] = "inlier"
        train = in_set.sample(frac=0.9)
        test = in_set[~in_set.image_path.isin(train.image_path)]
        
        #Add small novel examples to test
        novel_set = self.raw_df[self.raw_df.label.isin([8,9])]
        novel_set = novel_set.groupby("label").apply(lambda x: x.head(self.config["novel_class_examples"]))
        novel_set["outlier"] = "novel"
        novel_set["image_corrupt"] = False
        
        #label swap within train
        labels_to_corrupt = train.groupby("label").apply(lambda x: x.sample(frac=self.config["proportion_switch"]))
        labels_to_corrupt["observed_label"] = labels_to_corrupt.label.apply(lambda x: np.random.choice(range(8)))
        labels_to_corrupt["outlier"] = "label_swap"
        labels_to_corrupt["image_corrupt"] = False
        
        train = train[~train.image_path.isin(labels_to_corrupt.image_path)]
        #Image corrupt in train
        train["image_corrupt"] = np.random.choice([False, True], train.shape[0], p=[1-self.config["proportion_image_corrupt"], self.config["proportion_image_corrupt"]])
        train.loc[train["image_corrupt"] == True,"outlier"] = "corrupted"        
        train = pd.concat([train, labels_to_corrupt])
        
        #label swap within test
        labels_to_corrupt = test.groupby("label").apply(lambda x: x.sample(frac=self.config["proportion_switch"]))
        labels_to_corrupt["observed_label"] = labels_to_corrupt.label.apply(lambda x: np.random.choice(range(8)))
        labels_to_corrupt["outlier"] = "label_swap"
        labels_to_corrupt["image_corrupt"] = False
        
        test = test[~test.image_path.isin(labels_to_corrupt.image_path)]
        
        #image corrupt in test
        test["image_corrupt"] = np.random.choice([False, True], test.shape[0], p=[1-self.config["proportion_image_corrupt"], self.config["proportion_image_corrupt"]])
        test.loc[test["image_corrupt"] == True,"outlier"] = "corrupted"
        test = pd.concat([test, labels_to_corrupt])        
        test = pd.concat([test,novel_set])
        
        return train, test
    
    def setup(self, stage=None):
        self.raw_df = self.download_mnist()
        self.train, self.test = self.corrupt_and_split()
        self.train_ds = mnist_dataset(self.train)
        self.val_ds = mnist_dataset(self.test)
        self.num_classes = len(np.unique(self.train.label))
        
    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
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
        self.config = config
        self.log = log
        if self.log:
            self.comet_experiment = CometLogger(project_name="DeepTreeAttention", workspace=self.config["comet_workspace"],auto_output_logging = "simple")
            self.comet_experiment.experiment.add_tag("simulation")
            self.comet_experiment.experiment.log_parameters(self.config)
    
    def generate_data(self):
        """Simulation data from RAW MNIST dataset"""
        self.data_module = simulation_data(config=self.config)
        self.data_module.setup()
    
    def create_model(self):
        self.model = autoencoder(bands=1, classes=self.data_module.num_classes, config=self.config, comet_logger=self.comet_experiment)
        
    def train(self):
        """Train a neural network arch"""
        #Create trainer
        with self.comet_experiment.experiment.context_manager("classification_only"):
            self.model.config["classification_loss_scalar"] = 1
            self.model.config["autoencoder_loss_scalar"] = 0         
            self.trainer = Trainer(
                gpus=self.config["gpus"],
                fast_dev_run=self.config["fast_dev_run"],
                max_epochs=self.config["classifier_epochs"],
                accelerator=self.config["accelerator"],
                checkpoint_callback=False,
                logger=self.comet_experiment)
            
            self.trainer.fit(self.model, datamodule=self.data_module)
            
        with self.comet_experiment.experiment.context_manager("autoencoder_only"):  
            self.model.config["classification_loss_scalar"] = 0
            self.model.config["autoencoder_loss_scalar"] = 1
            self.trainer = Trainer(
                gpus=self.config["gpus"],
                fast_dev_run=self.config["fast_dev_run"],
                max_epochs=self.config["autoencoder_epochs"],
                accelerator=self.config["accelerator"],
                checkpoint_callback=False,
                logger=self.comet_experiment)

            #freeze classification and below layers
            for x in self.model.parameters():
                x.requires_grad = False
            
            for x in self.model.decoder_block1.parameters():
                x.requires_grad = True
            
            for x in self.model.decoder_block2.parameters():
                x.requires_grad = True
            
            for x in self.model.decoder_block3.parameters():
                x.requires_grad = True
                        
            self.trainer.fit(self.model, datamodule=self.data_module)
            
    def predict_validation(self):
        """Generate labels and predictions for validation data_loader"""
        observed_y = []
        yhat = []
        y = []
        autoencoder_loss = []
        sample_ids = []
        classification_bottleneck = []
        
        self.model.eval()
        
        for batch in self.model.val_dataloader():
            index, images, observed_labels, labels = batch
            observed_y.append(observed_labels.numpy())
            y.append(labels)
            sample_ids.append(index)
            if next(self.model.parameters()).is_cuda:
                images = images.cuda()
            with torch.no_grad():
                for image in images:
                    image_yhat, classification_yhat, features = self.model(image.unsqueeze(0))
                    yhat.append(classification_yhat)
                    loss = F.mse_loss(image_yhat, image)    
                    autoencoder_loss.append(loss.numpy())
                    classification_bottleneck.append(features.cpu().numpy())                    
           
        yhat = np.concatenate(yhat)
        yhat = np.argmax(yhat, 1)
        y = np.concatenate(y)
        sample_ids = np.concatenate(sample_ids)
        observed_y = np.concatenate(observed_y)
        autoencoder_loss = np.asarray(autoencoder_loss)
        
        #Create a single array
        self.classification_bottleneck = np.concatenate(classification_bottleneck)
        
        #look up sample ids
        outlier_class = self.data_module.test.outlier.iloc[sample_ids].astype('category').cat.codes.astype(int).values
        
        #plot different sets
        layerplot_vis = visualize.plot_2d_layer(features=self.classification_bottleneck, labels=observed_y, use_pca=True)
        self.comet_experiment.experiment.log_figure(figure=layerplot_vis, figure_name="classification_bottleneck_labels", step=self.model.current_epoch)        
        
        layerplot_vis = visualize.plot_2d_layer(features=self.classification_bottleneck, labels=outlier_class, use_pca=True, size_weights=outlier_class+1)        
        self.comet_experiment.experiment.log_figure(figure=layerplot_vis, figure_name="classification_bottleneck_outliers", step=self.model.current_epoch)

        results = pd.DataFrame({"test_index":sample_ids,"label":y,"observed_label": observed_y,"predicted_label":yhat, "autoencoder_loss": autoencoder_loss})        
    
        return results
    
    def evaluate(self):
        """Generate evaluation statistics for outlier detection"""
        results = self.predict_validation()
        results["outlier"] = results.test_index.apply(lambda x: self.data_module.test.iloc[x].outlier)
        results["image_corrupt"] = results.test_index.apply(lambda x: self.data_module.test.iloc[x].image_corrupt)        
        outlier_detection_loss = outlier.autoencoder_outliers(results, outlier_threshold=self.config["outlier_threshold"], experiment=self.comet_experiment.experiment)
        outlier_detection_distance = outlier.distance_outliers(results, self.classification_bottleneck, labels=results.observed_label, threshold=self.config["distance_threshold"], experiment=self.comet_experiment.experiment)
        novel_detection = outlier.novel_detection(results, self.classification_bottleneck, experiment=self.comet_experiment.experiment)
        results = pd.concat([outlier_detection_loss, outlier_detection_distance, novel_detection])
        
        return results

def run(ID, config):
    """A single run of the simulation"""
    #Set a couple dask env variables
    sim = simulator(config)    
    sim.generate_data()
    sim.create_model()
    sim.train()
    results = sim.evaluate()
    results["simulation_id"] = ID
    
    #Cleanup tmpdir
    sim.data_module.tmpdir.cleanup()
    
    return results