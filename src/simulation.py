#MNSIT simulation
import os
import numpy as np
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from src.models.simulation import autoencoder
from src import visualize
from skimage import io
import torch
import torchvision
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F

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
        image = io.imread(image_path)
        observed_label = self.annotations.observed_label.iloc[index]      
        true_label = self.annotations.true_label.iloc[index]      
        
        image = self.transforms(image)
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
            if class_labels[label] < self.config["samples"]:
                class_labels[label] = class_labels[label] + 1
                labels.append(label)
                fname = "{}/data/simulation/{}_{}.png".format(ROOT, label,class_labels[label])
                image_paths.append(fname)
                image.save(fname)
            
        df = pd.DataFrame({"image_path":image_paths, "label":labels})
        
        return df 
    
    def corrupt_and_split(self):
        """Switch labels of some within sample class and add some novel classes"""
        self.raw_df["true_label"] = self.raw_df["label"]
        self.raw_df["observed_label"] = self.raw_df["true_label"]
        
        novel_set = self.raw_df[self.raw_df.label.isin([8,9])]
        novel_set = novel_set.groupby("label").apply(lambda x: x.head(self.config["novel_class_examples"]))
        novel_set["outlier"] = "novel"
        
        in_set = self.raw_df[~self.raw_df.label.isin([8,9])]
        
        #label swap within in set
        labels_to_corrupt = in_set.groupby("label").apply(lambda x: x.sample(frac=self.config["proportion_switch"]))
        labels_to_corrupt["observed_label"] = labels_to_corrupt.label.apply(lambda x: np.random.choice(range(8)))
        labels_to_corrupt["outlier"] = "label_swap"
        
        uncorrupted_labels = in_set[~in_set.image_path.isin(labels_to_corrupt.image_path)]
        uncorrupted_labels["outlier"] = False
        
        self.corrupted_data = pd.concat([uncorrupted_labels, labels_to_corrupt])
        
        train = self.corrupted_data.groupby("observed_label").sample(frac = 0.9)
        test = self.corrupted_data[~self.corrupted_data.image_path.isin(train.image_path)]  
        
        #add novel to just test
        test = pd.concat([test, novel_set])
        
        return train, test
    
    def setup(self, stage=None):
        self.raw_df = self.download_mnist()
        self.train, self.test = self.corrupt_and_split()
        self.train_ds = mnist_dataset(self.train)
        self.val_ds = mnist_dataset(self.test)
        self.num_classes = len(np.unique(self.test.label))
        
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
        self.config = config
        self.log = log
        if self.log:
            self.comet_experiment = CometLogger(project_name="DeepTreeAttention", workspace=self.config["comet_workspace"],auto_output_logging = "simple")
            self.comet_experiment.experiment.add_tag("simulation")
            self.comet_experiment.experiment.log_parameters(self.config)
    
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
    
    def generate_plots(self):
        """At the end of each epoch trigger the dataset to collect intermediate activation for plotting"""
        #plot 2d projection layer
        epoch_labels = []
        vis_epoch_activations = []
        encoder_epoch_activations = []
        sample_ids = []
        for batch in self.data_module.val_dataloader():
            index, images, observed_labels, true_labels  = batch
            epoch_labels.append(observed_labels)
            sample_ids.append(index)
            
            #trigger activation hook
            if next(self.model.parameters()).is_cuda:
                image = images.cuda()
            else:
                image = images
            
            pred = self.model(image)
            vis_epoch_activations.append(self.model.vis_activation["vis_layer"].cpu())
            encoder_epoch_activations.append(self.model.vis_activation["encoder_block3"].cpu())

        #Create a single array
        epoch_labels = np.concatenate(epoch_labels)
        vis_epoch_activations = torch.tensor(np.concatenate(vis_epoch_activations))
        encoder_epoch_activations = torch.tensor(np.concatenate(encoder_epoch_activations))
        sample_ids = np.concatenate(sample_ids)
        
        #look up sample ids
        outlier_class = self.data_module.test.outlier.loc[sample_ids]
        
        #plot different sets
        layerplot_vis = visualize.plot_2d_layer(vis_epoch_activations, epoch_labels)
        self.comet_experiment.experiment.log_figure(figure=layerplot_vis, figure_name="2d_vis_projection_labels", step=self.current_epoch)        
        
        layerplot_vis = visualize.plot_2d_layer(vis_epoch_activations, outlier_class)        
        self.comet_experiment.experiment.log_figure(figure=layerplot_vis, figure_name="2d_vis_projection_labels", step=self.current_epoch)

        layerplot_encoder = visualize.plot_2d_layer(encoder_epoch_activations, epoch_labels, use_pca=True)
        self.comet_experiment.experiment.log_figure(figure=layerplot_encoder, figure_name="PCA_encoder_projection_labels", step=self.current_epoch)
        layerplot_encoder = visualize.plot_2d_layer(encoder_epoch_activations, outlier_class, use_pca=True)
        self.comet_experiment.experiment.log_figure(figure=layerplot_encoder, figure_name="PCA_encoder_projection_outliers", step=self.current_epoch)
        
    def outlier_detection(self, results):
        """Given a set of predictions, label outliers"""
        threshold = results.autoencoder_loss.quantile(self.config["outlier_threshold"])
        print("Reconstruction threshold is {}".format(threshold))
        results["outlier"] = results.autoencoder_loss > threshold
        
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
        y = np.concatenate(y)
        observed_y = np.concatenate(observed_y)
        autoencoder_loss = np.asarray(autoencoder_loss)
        
        results = pd.DataFrame({"true_label":y,"observed_label": observed_y,"predicted_label":yhat, "autoencoder_loss": autoencoder_loss})
        results = self.outlier_detection(results)
        
        if self.log:
            self.comet_experiment.experiment.log_table("results.csv",results)
        
        #Mean Proportion of true classes are correct
        mean_accuracy = results[~results.true_label.isin([8,9])].groupby("true_label").apply(lambda x: x.true_label == x.predicted_label).mean()
        
        if self.log:
            self.comet_experiment.experiment.log_metric(name="Mean accuracy", value=mean_accuracy)
        
        true_outliers = results[~(results.true_label == results.observed_label)]
        #inset data does not have class 8 ir 9
        inset = true_outliers[~true_outliers.true_label.isin([8,9])]
        outlier_accuracy = sum(inset.outlier)/inset.shape[0]
        outlier_precision = sum(inset.outlier)/results.filter(~results.true_label.isin([8,9])).shape[0]
        
        if self.log:
            self.comet_experiment.experiment.log_metric("outlier_accuracy", outlier_accuracy)
            self.comet_experiment.experiment.log_metric("outlier_precision", outlier_precision)
            
        #TODO 
        #label_switching = self.label_switching(df, self.data_module.true_state)
        
        return pd.DataFrame({"outlier_accuracy": [outlier_accuracy], "outlier_precision": [outlier_precision], "classification_accuracy": [mean_accuracy]})
        
def run(ID, config):
    #Set a couple dask env variables
    sim = simulator(config)    
    sim.generate_data()
    sim.create_model()
    sim.train()
    sim.generate_plots()
    results = sim.evaluate()
    results["simulation_id"] = ID
    return results
    