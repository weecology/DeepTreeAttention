#Lightning Data Module
from . import __file__
import pandas as pd
import torch
import copy
from torch.nn import functional as F

from src import fixmatch, visualize, data, semi_supervised
from src.models import multi_stage

class TreeModel(multi_stage.MultiStage):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self, supervised_train, supervised_test, taxonomic_csv, config=None, client=None):
        super(TreeModel, self).__init__(train_df=supervised_train, test_df=supervised_test, config=config, taxonomic_csv=taxonomic_csv)
        
        # Unsupervised versus supervised loss weight
        self.alpha = torch.nn.Parameter(torch.tensor(self.config["semi_supervised"]["alpha"], dtype=float), requires_grad=False)
        if self.config["semi_supervised"]["semi_supervised_train"] is None:
            self.semi_supervised_train = semi_supervised.create_dataframe(config, client=client)
        else:
            self.semi_supervised_train = pd.read_csv(self.config["semi_supervised"]["semi_supervised_train"])
        
        if self.config["semi_supervised"]["max_samples_per_class"] is not None:
            individuals_to_keep = self.semi_supervised_train.groupby("taxonID").apply(lambda x: x.drop_duplicates(subset="individual").sample(frac=1).head(n=config["semi_supervised"]["max_samples_per_class"])).individual.values
            self.semi_supervised_train = self.semi_supervised_train[self.semi_supervised_train.individual.isin(individuals_to_keep)].reset_index(drop=True)            
            
        self.supervised_train = supervised_train
        self.supervised_test = supervised_test       
        
    def train_dataloader(self):
        ## Labeled data
        self.data_loader = super(TreeModel, self).train_dataloader()
        
        ## Unlabeled data
        semi_supervised_config = copy.deepcopy(self.config)
        semi_supervised_config["crop_dir"] = semi_supervised_config["semi_supervised"]["crop_dir"]
        semi_supervised_config["preload_images"] = semi_supervised_config["semi_supervised"]["preload_images"]

        unlabeled_ds = fixmatch.FixmatchDataset(
            df=self.semi_supervised_train,
            config=semi_supervised_config
        )
        
        self.unlabeled_data_loader = torch.utils.data.DataLoader(
            unlabeled_ds,
            batch_size=self.config["semi_supervised"]["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )           
        
        return {"labeled":self.data_loader, "unlabeled": self.unlabeled_data_loader}
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        """Train on a loaded dataset
        """
        # Labeled data
        individual, inputs, y = batch["labeled"][optimizer_idx]
        labeled_images = inputs["HSI"]
        y_hat = self.models[optimizer_idx].forward(labeled_images)
        supervised_loss = F.cross_entropy(y_hat, y)   
        
        ## Unlabeled data - Weak Augmentation
        individual, inputs = batch["unlabeled"]
        unlabeled_images = inputs["Weak"]
        
        #Combine labeled and unlabeled data to preserve batchnorm
        self.models[optimizer_idx].eval()        
        logit_weak = self.models[optimizer_idx].forward(unlabeled_images)  
        self.models[optimizer_idx].train()        
        prob_weak = F.softmax(logit_weak, dim=1)
        
        # Unlabeled data - Strong Augmentation
        images = inputs["Strong"]
        self.models[optimizer_idx].eval()
        logit_strong = self.models[optimizer_idx].forward(images)
        self.models[optimizer_idx].train()
        
        #Only select those labels greater than threshold
        p_pseudo_label, pseudo_label = torch.max(prob_weak.detach(), dim=-1)
        threshold_mask = p_pseudo_label.ge(self.config["semi_supervised"]["fixmatch_threshold"]).float()
        pseudo_loss = F.cross_entropy(logit_strong, pseudo_label, reduction="none")
        pseudo_loss = (pseudo_loss * threshold_mask).mean()
        self.unlabeled_samples_count = self.unlabeled_samples_count + sum(threshold_mask)
        
        self.log("Unlabeled mean training confidence",p_pseudo_label.mean())            
        self.log("supervised_loss_{}".format(optimizer_idx),supervised_loss)
        self.log("unsupervised_loss_{}".format(optimizer_idx), pseudo_loss)
        
        if self.current_epoch > 20:
            loss = supervised_loss + (self.alpha * pseudo_loss) 
        else:
            loss = supervised_loss
        
        return loss
    
    def fixmatch_dataloader(self, df):
        """Validation data loader only includes labeled data"""
        
        semi_supervised_config = copy.deepcopy(self.config)
        semi_supervised_config["crop_dir"] = semi_supervised_config["semi_supervised"]["crop_dir"]
        semi_supervised_config["preload_images"] = semi_supervised_config["semi_supervised"]["preload_images"]

        ds = fixmatch.FixmatchDataset(
            df=df,
            config=semi_supervised_config
        )
        
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=True,
            num_workers=self.config["workers"],
        )
        return data_loader  
    
    def fixmatch_step(self, batch, batch_idx, level):
        ## Unlabeled data - Weak Augmentation
        individual, inputs = batch
        unlabeled_images = inputs["Weak"]
        
        #Combine labeled and unlabeled data to preserve batchnorm
        self.models[level].eval()
        logit_weak = self.models[level].forward(unlabeled_images)  
        prob_weak = F.softmax(logit_weak, dim=1)
        
        # Unlabeled data - Strong Augmentation
        images = inputs["Strong"]
        logit_strong = self.models[level].forward(images)
        prob_strong = F.softmax(logit_strong, dim=1)
        self.models[level].train()
        
        return individual, prob_strong, prob_weak

    def on_train_start(self):
        """On training start log a sample set of data to visualize spectra"""
        
        self.unlabeled_samples_count = 0
        
        individual_to_keep = self.semi_supervised_train.groupby("taxonID").apply(lambda x: x.sample(frac=1).head(n=4)).individual
        self.train_viz_df = self.semi_supervised_train[self.semi_supervised_train.individual.isin(individual_to_keep)].reset_index(drop=True)
        self.trainer.logger.experiment.log_table("train_viz.csv", self.train_viz_df)
        self.train_viz_dl = self.fixmatch_dataloader(df=self.train_viz_df)
        
        for batch in self.train_viz_dl:
            individual, inputs = batch
            weak = inputs["Weak"]
            strong = inputs["Strong"]
            for level in range(self.levels):            
                spectra = torch.stack([weak[0],strong[0]]).mean([1,3, 4]).cpu().numpy()
                pd.DataFrame(spectra.T).plot()
                self.trainer.logger.experiment.log_figure("{}_spectra_level_{}".format(individual, level))

        for i, batch in enumerate(self.train_viz_dl):
            for level in range(self.levels):
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                individual, strong, weak = self.fixmatch_step(batch, i, level)
                figure = visualize.visualize_consistency(strong, weak)
                self.trainer.logger.experiment.log_figure("{}_softmax_pretrain_level_{}".format(individual, level))

    def on_train_epoch_end(self):
        for i, batch in enumerate(self.train_viz_dl):
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            individual, strong, weak = self.predict_step(batch, i)
            figure = visualize.visualize_consistency(strong, weak)
            self.trainer.logger.experiment.log_figure("{}_softmax".format(individual))  

    def on_train_epoch_end(self):
        self.log("unlabeled_samples",self.unlabeled_samples_count)
        
            