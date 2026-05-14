# Unified single-model hierarchical classifier.
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics

from src.data import TreeDataset


CONIFER_TAXA = {"PICL", "PIEL", "PITA"}


class SharedHSIEncoder(nn.Module):
    """Lightweight shared encoder used for all years."""

    def __init__(self, bands: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=(2, 3))
        return self.proj(x)


class UnifiedHierarchicalHead(nn.Module):
    """Single model with year-aware aggregation + metadata fusion + multi-head outputs."""

    def __init__(
        self,
        bands: int,
        n_years: int,
        n_sites: int,
        n_species: int,
        level_dims: list[int],
        embed_dim: int = 128,
        site_embed_dim: int = 16,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_years = max(1, n_years)
        self.unknown_site_index = n_sites
        self.encoder = SharedHSIEncoder(bands=bands, embed_dim=embed_dim)
        self.year_embedding = nn.Embedding(self.n_years, embed_dim)
        self.time_attention = nn.Linear(embed_dim, 1)
        self.site_embedding = nn.Embedding(n_sites + 1, site_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + site_embed_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.species_head = nn.Linear(fusion_hidden_dim, n_species)
        self.level_heads = nn.ModuleList(
            [nn.Linear(fusion_hidden_dim, d) if d > 0 else nn.Identity() for d in level_dims]
        )
        self.level_dims = level_dims

    def forward(self, images: list[torch.Tensor], site_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        year_feats = []
        masks = []
        for year_idx, x in enumerate(images):
            feat = self.encoder(x)
            year_tensor = torch.full(
                (feat.shape[0],),
                min(year_idx, self.n_years - 1),
                device=feat.device,
                dtype=torch.long,
            )
            feat = feat + self.year_embedding(year_tensor)
            mask = (x.abs().sum(dim=(1, 2, 3)) > 0).float()
            year_feats.append(feat)
            masks.append(mask)

        feat_stack = torch.stack(year_feats, dim=1)
        mask_stack = torch.stack(masks, dim=1)
        attn_logits = self.time_attention(feat_stack).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask_stack <= 0, -1e9)
        attn = torch.softmax(attn_logits, dim=1) * mask_stack
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (attn.unsqueeze(-1) * feat_stack).sum(dim=1)

        if site_idx is None:
            site_idx = torch.full(
                (pooled.shape[0],),
                self.unknown_site_index,
                device=pooled.device,
                dtype=torch.long,
            )
        else:
            site_idx = site_idx.to(pooled.device).long().clamp(0, self.unknown_site_index)
        site_feat = self.site_embedding(site_idx)

        fused = self.fusion(torch.cat([pooled, site_feat], dim=1))
        outputs = {"species": self.species_head(fused)}
        for level_idx, head in enumerate(self.level_heads):
            key = f"level_{level_idx}"
            if self.level_dims[level_idx] <= 0:
                outputs[key] = torch.empty((fused.shape[0], 0), device=fused.device)
            else:
                outputs[key] = head(fused)
        return outputs


class MultiStage(LightningModule):
    """
    Unified hierarchical model in one checkpoint.
    Keeps the old MultiStage API used by training/inference code paths.
    """

    def __init__(
        self,
        train_df=None,
        test_df=None,
        crowns=None,
        config=None,
        train_mode=True,
        years=None,
        classes=None,
        species_label_dict=None,
        level_label_dicts=None,
        n_sites=0,
    ):
        super().__init__()
        self.config = config or {}
        self.crowns = crowns
        self.train_df = train_df.copy() if train_df is not None else pd.DataFrame()
        self.test_df = test_df.copy() if test_df is not None else pd.DataFrame()

        if not self.train_df.empty and "individual" not in self.train_df.columns and "individualID" in self.train_df.columns:
            self.train_df["individual"] = self.train_df["individualID"]
        if not self.test_df.empty and "individual" not in self.test_df.columns and "individualID" in self.test_df.columns:
            self.test_df["individual"] = self.test_df["individualID"]

        if not self.train_df.empty:
            self.years = sorted(self.train_df.tile_year.unique())
            self.classes = int(self.train_df.label.nunique())
            self.species_label_dict = (
                self.train_df[["taxonID", "label"]].drop_duplicates().set_index("taxonID")["label"].to_dict()
            )
            self.level_label_dicts = self._build_level_maps()
            if "site" in self.train_df.columns:
                n_sites = int(self.train_df["site"].max()) + 1
        else:
            self.years = years or []
            self.classes = int(classes or 0)
            self.species_label_dict = species_label_dict or {}
            self.level_label_dicts = level_label_dicts or [{"PIPA2": 0, "OTHER": 1}, {"CONIFER": 0, "BROADLEAF": 1}, {}, {}, {}]
        self.index_to_label = {v: k for k, v in self.species_label_dict.items()}
        self.label_to_taxonIDs = [{v: k for k, v in d.items()} for d in self.level_label_dicts]
        level_dims = [len(x) for x in self.level_label_dicts]

        self.model = UnifiedHierarchicalHead(
            bands=int(self.config["bands"]),
            n_years=max(1, len(self.years)),
            n_sites=n_sites,
            n_species=self.classes,
            level_dims=level_dims,
            embed_dim=int(self.config.get("hier_embed_dim", 128)),
            site_embed_dim=int(self.config.get("hier_site_embed_dim", 16)),
            fusion_hidden_dim=int(self.config.get("hier_fusion_dim", 256)),
        )

        self._val_epoch_outputs = []
        self.train_dataset = None
        self.test_dataset = None
        self._make_training_views()

        if not self.train_df.empty:
            self.train_dataset = TreeDataset(df=self.train_df, config=self.config, train=True)
            self.test_dataset = TreeDataset(df=self.test_df, config=self.config, train=True)
            counts = self.train_df["label"].value_counts().sort_index()
            counts = counts.reindex(range(self.classes), fill_value=1).astype(float)
            inv = 1.0 / counts.to_numpy()
            inv = inv / np.max(inv)
            min_w = float(self.config.get("min_loss_weight", 0.05))
            inv = np.clip(inv, min_w, None)
            prior = counts.to_numpy() / counts.to_numpy().sum()
        else:
            inv = np.ones(max(1, self.classes), dtype=float)
            prior = np.ones(max(1, self.classes), dtype=float) / max(1, self.classes)
        self.register_buffer("species_loss_weight", torch.tensor(inv, dtype=torch.float32))
        self.register_buffer("species_log_prior", torch.tensor(np.log(prior + 1e-12), dtype=torch.float32))

        micro = torchmetrics.Accuracy(task="multiclass", num_classes=self.classes, average="micro")
        macro = torchmetrics.Accuracy(task="multiclass", num_classes=self.classes, average="macro")
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy": micro, "Macro Accuracy": macro})
        self.save_hyperparameters(ignore=["train_df", "test_df", "crowns"])

    def _build_level_maps(self) -> list[dict[str, int]]:
        taxa = sorted(self.species_label_dict.keys())
        level0 = {"PIPA2": 0, "OTHER": 1}
        level1 = {"CONIFER": 0, "BROADLEAF": 1}

        broadleaf = [t for t in taxa if t not in CONIFER_TAXA and t != "PIPA2" and not t.startswith("QU")]
        level2 = {t: i for i, t in enumerate(broadleaf)}
        level2["OAK"] = len(level2)

        conifer = [t for t in taxa if t in CONIFER_TAXA]
        level3 = {t: i for i, t in enumerate(conifer)}

        oak = [t for t in taxa if t.startswith("QU")]
        level4 = {t: i for i, t in enumerate(oak)}
        return [level0, level1, level2, level3, level4]

    def _make_training_views(self):
        if self.train_df.empty or self.test_df.empty:
            self.level_0_train = pd.DataFrame()
            self.level_1_train = pd.DataFrame()
            self.level_2_train = pd.DataFrame()
            self.level_3_train = pd.DataFrame()
            self.level_4_train = pd.DataFrame()
            self.level_0_test = pd.DataFrame()
            self.level_1_test = pd.DataFrame()
            self.level_2_test = pd.DataFrame()
            self.level_3_test = pd.DataFrame()
            self.level_4_test = pd.DataFrame()
            return

        self.level_0_train = self.train_df.copy()
        self.level_0_train["taxonID"] = self.level_0_train["taxonID"].where(
            self.level_0_train["taxonID"] == "PIPA2", "OTHER"
        )
        self.level_0_test = self.test_df.copy()
        self.level_0_test["taxonID"] = self.level_0_test["taxonID"].where(
            self.level_0_test["taxonID"] == "PIPA2", "OTHER"
        )

        self.level_1_train = self.train_df.copy()
        self.level_1_train["taxonID"] = np.where(
            self.level_1_train["taxonID"].isin(CONIFER_TAXA), "CONIFER", "BROADLEAF"
        )
        self.level_1_test = self.test_df.copy()
        self.level_1_test["taxonID"] = np.where(
            self.level_1_test["taxonID"].isin(CONIFER_TAXA), "CONIFER", "BROADLEAF"
        )

        self.level_2_train = self.train_df.copy()
        self.level_2_train = self.level_2_train[
            ~self.level_2_train["taxonID"].isin(CONIFER_TAXA.union({"PIPA2"}))
        ].copy()
        self.level_2_train.loc[self.level_2_train["taxonID"].str.startswith("QU"), "taxonID"] = "OAK"
        self.level_2_test = self.test_df.copy()
        self.level_2_test = self.level_2_test[
            ~self.level_2_test["taxonID"].isin(CONIFER_TAXA.union({"PIPA2"}))
        ].copy()
        self.level_2_test.loc[self.level_2_test["taxonID"].str.startswith("QU"), "taxonID"] = "OAK"

        self.level_3_train = self.train_df[self.train_df["taxonID"].isin(CONIFER_TAXA)].copy()
        self.level_3_test = self.test_df[self.test_df["taxonID"].isin(CONIFER_TAXA)].copy()
        self.level_4_train = self.train_df[self.train_df["taxonID"].str.startswith("QU")].copy()
        self.level_4_test = self.test_df[self.test_df["taxonID"].str.startswith("QU")].copy()

    def _loader_kwargs(self):
        workers = int(self.config.get("workers") or 0)
        kw = {"num_workers": workers}
        if workers > 0:
            kw["persistent_workers"] = True
        return kw

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            **self._loader_kwargs(),
        )

    def predict_dataloader(self, ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["predict_batch_size"],
            shuffle=False,
            **self._loader_kwargs(),
        )

    def _hierarchy_targets(self, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        taxa = [self.index_to_label[int(x)] for x in labels.detach().cpu().tolist()]
        device = labels.device
        out = {}

        level0 = [self.level_label_dicts[0]["PIPA2"] if t == "PIPA2" else self.level_label_dicts[0]["OTHER"] for t in taxa]
        out["level_0"] = torch.tensor(level0, dtype=torch.long, device=device)

        level1 = [self.level_label_dicts[1]["CONIFER"] if t in CONIFER_TAXA else self.level_label_dicts[1]["BROADLEAF"] for t in taxa]
        out["level_1"] = torch.tensor(level1, dtype=torch.long, device=device)

        level2 = []
        for t in taxa:
            if t in CONIFER_TAXA or t == "PIPA2":
                level2.append(-100)
            elif t.startswith("QU"):
                level2.append(self.level_label_dicts[2]["OAK"])
            else:
                level2.append(self.level_label_dicts[2].get(t, -100))
        out["level_2"] = torch.tensor(level2, dtype=torch.long, device=device)

        level3 = [self.level_label_dicts[3].get(t, -100) for t in taxa]
        out["level_3"] = torch.tensor(level3, dtype=torch.long, device=device)

        level4 = [self.level_label_dicts[4].get(t, -100) for t in taxa]
        out["level_4"] = torch.tensor(level4, dtype=torch.long, device=device)
        return out

    def _species_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tau = float(self.config.get("logit_adjustment_tau", 1.0))
        adjusted = logits + (tau * self.species_log_prior).to(logits.device)
        return F.cross_entropy(adjusted, labels, weight=self.species_loss_weight.to(logits.device))

    def _aux_losses(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        targets = self._hierarchy_targets(labels)
        total = torch.zeros((), device=labels.device)
        for idx in range(5):
            key = f"level_{idx}"
            logits = outputs[key]
            if logits.numel() == 0:
                continue
            target = targets[key]
            if not torch.any(target != -100):
                continue
            loss = F.cross_entropy(logits, target, ignore_index=-100)
            weight = float(self.config.get(f"hier_level_{idx}_weight", 1.0))
            total = total + weight * loss
        return total

    def training_step(self, batch, batch_idx):
        _, inputs, labels = batch
        outputs = self.model(inputs["HSI"], inputs.get("site"))
        species_loss = self._species_loss(outputs["species"], labels)
        aux_loss = self._aux_losses(outputs, labels)
        alpha = float(self.config.get("hier_aux_weight", 0.3))
        loss = species_loss + alpha * aux_loss
        self.log("train_species_loss", species_loss, on_step=False, on_epoch=True)
        self.log("train_aux_loss", aux_loss, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, inputs, labels = batch
        outputs = self.model(inputs["HSI"], inputs.get("site"))
        species_loss = self._species_loss(outputs["species"], labels)
        aux_loss = self._aux_losses(outputs, labels)
        alpha = float(self.config.get("hier_aux_weight", 0.3))
        loss = species_loss + alpha * aux_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = F.softmax(outputs["species"], dim=1)
        metrics = self.metrics(probs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self._val_epoch_outputs.append({"probs": probs.detach().cpu(), "labels": labels.detach().cpu()})
        return loss

    def on_validation_epoch_start(self):
        self._val_epoch_outputs = []

    def on_validation_epoch_end(self):
        if not self._val_epoch_outputs:
            return
        probs = torch.cat([x["probs"] for x in self._val_epoch_outputs], dim=0)
        labels = torch.cat([x["labels"] for x in self._val_epoch_outputs], dim=0)
        preds = torch.argmax(probs, dim=1)

        epoch_micro = torchmetrics.functional.accuracy(
            preds=preds, target=labels, task="multiclass", num_classes=self.classes, average="micro"
        )
        epoch_macro = torchmetrics.functional.accuracy(
            preds=preds, target=labels, task="multiclass", num_classes=self.classes, average="macro"
        )
        self.log("Epoch Micro Accuracy", epoch_micro, on_step=False, on_epoch=True)
        self.log("Epoch Macro Accuracy", epoch_macro, on_step=False, on_epoch=True)
        self._val_epoch_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr", 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.75,
            patience=8,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            eps=1e-08,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def predict_step(self, batch, batch_idx):
        individual, inputs = batch
        outputs = self.model(inputs["HSI"], inputs.get("site"))
        return {
            "individual": np.asarray(individual),
            "species_probs": F.softmax(outputs["species"], dim=1).detach().cpu().numpy(),
            "level_0_probs": F.softmax(outputs["level_0"], dim=1).detach().cpu().numpy(),
            "level_1_probs": F.softmax(outputs["level_1"], dim=1).detach().cpu().numpy(),
            "level_2_probs": F.softmax(outputs["level_2"], dim=1).detach().cpu().numpy() if outputs["level_2"].numel() else None,
            "level_3_probs": F.softmax(outputs["level_3"], dim=1).detach().cpu().numpy() if outputs["level_3"].numel() else None,
            "level_4_probs": F.softmax(outputs["level_4"], dim=1).detach().cpu().numpy() if outputs["level_4"].numel() else None,
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        kwargs.setdefault("weights_only", False)
        return super(MultiStage, cls).load_from_checkpoint(checkpoint_path, **kwargs)

    def gather_predictions(self, predict_df):
        """Aggregate predict outputs and expose level columns for compatibility."""
        individuals = []
        species_probs = []
        level_probs = {f"level_{i}": [] for i in range(5)}
        for output in predict_df:
            individuals.extend(list(output["individual"]))
            species_probs.append(output["species_probs"])
            for i in range(5):
                p = output[f"level_{i}_probs"]
                if p is not None:
                    level_probs[f"level_{i}"].append(p)

        species_probs = np.vstack(species_probs)
        results = pd.DataFrame(
            {
                "individual": np.asarray(individuals),
                "pred_label_top1": np.argmax(species_probs, axis=1),
                "top1_score": np.max(species_probs, axis=1),
            }
        )
        results["pred_taxa_top1"] = results["pred_label_top1"].map(self.index_to_label)

        for i in range(5):
            key = f"level_{i}"
            if level_probs[key]:
                probs = np.vstack(level_probs[key])
                results[f"pred_label_top1_level_{i}"] = np.argmax(probs, axis=1)
                results[f"top1_score_level_{i}"] = np.max(probs, axis=1)
                inv = self.label_to_taxonIDs[i]
                results[f"pred_taxa_top1_level_{i}"] = results[f"pred_label_top1_level_{i}"].map(inv)
            else:
                results[f"pred_label_top1_level_{i}"] = np.nan
                results[f"top1_score_level_{i}"] = np.nan
                results[f"pred_taxa_top1_level_{i}"] = None

        return results

    def ensemble(self, results):
        """Single-pass species head output, while keeping legacy column names."""
        out = results.copy()
        if "pred_taxa_top1" in out.columns:
            out["ensembleTaxonID"] = out["pred_taxa_top1"]
            out["ens_label"] = out["pred_label_top1"]
            out["ens_score"] = out["top1_score"]
            return out

        # Fallback for externally-built legacy tables.
        out["ensembleTaxonID"] = out.get("pred_taxa_top1_level_2")
        out["ens_score"] = out.get("top1_score_level_2")
        out["ens_label"] = out["ensembleTaxonID"].map(self.species_label_dict)
        return out

    def evaluation_scores(self, ensemble_df, experiment):
        ensemble_df = ensemble_df.drop_duplicates(subset=["individual"], keep="first")
        n_cls = len(self.species_label_dict)
        ed = ensemble_df.dropna(subset=["ens_label", "label"])
        ed = ed[
            (ed["ens_label"] >= 0)
            & (ed["ens_label"] < n_cls)
            & (ed["label"] >= 0)
            & (ed["label"] < n_cls)
        ]
        if ed.empty:
            return ensemble_df

        preds = torch.tensor(ed["ens_label"].values, dtype=torch.long)
        target = torch.tensor(ed["label"].values, dtype=torch.long)
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=preds,
            target=target,
            task="multiclass",
            num_classes=n_cls,
            average="none",
        )
        taxon_precision = torchmetrics.functional.precision(
            preds=preds,
            target=target,
            task="multiclass",
            num_classes=n_cls,
            average="none",
        )

        taxon_labels = sorted(self.species_label_dict.keys(), key=lambda t: self.species_label_dict[t])
        species_table = pd.DataFrame(
            {"taxonID": taxon_labels, "accuracy": taxon_accuracy, "precision": taxon_precision}
        )
        if experiment:
            experiment.log_metrics(species_table.set_index("taxonID").accuracy.to_dict(), prefix="accuracy")
            experiment.log_metrics(species_table.set_index("taxonID").precision.to_dict(), prefix="precision")

        if experiment and "siteID" in ed.columns:
            site_data_frame = []
            for name, group in ed.groupby("siteID"):
                g = group[
                    (group["ens_label"] >= 0)
                    & (group["ens_label"] < n_cls)
                    & (group["label"] >= 0)
                    & (group["label"] < n_cls)
                ]
                if g.empty:
                    continue
                site_micro = np.sum(g.ens_label.values == g.label.values) / len(g.ens_label.values)
                site_macro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(g["ens_label"].values, dtype=torch.long),
                    target=torch.tensor(g["label"].values, dtype=torch.long),
                    task="multiclass",
                    num_classes=n_cls,
                    average="macro",
                )
                experiment.log_metric("{}_macro".format(name), site_macro)
                experiment.log_metric("{}_micro".format(name), site_micro)
                row = pd.DataFrame({"Site": [name], "Micro Recall": [site_micro], "Macro Recall": [site_macro]})
                site_data_frame.append(row)
            if site_data_frame:
                site_data_frame = pd.concat(site_data_frame)
                experiment.log_table("site_results.csv", site_data_frame)

        return ensemble_df