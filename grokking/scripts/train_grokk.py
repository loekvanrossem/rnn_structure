from fractions import Fraction
import torch
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import IterableDataset
from datasets import AbstractDataset
from utils import combine_logs
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item

import numpy as np
import random
import scipy
from sklearn.cluster import AgglomerativeClustering


def generate_long_input(dataset, length, N):
    symbols = set(dataset.vocab2idx.values())
    inputs = []
    for _ in range(N):
        seq = random.sample(symbols, length)
        # dataset.encode(seq)
        inputs.append(seq)
    inputs = torch.tensor(inputs)

    return inputs


# def count_states(H, threshold):
#     dist = scipy.spatial.distance_matrix(H, H)
#     count = np.sum(dist < threshold)
#     n_datapoints = H.shape[0]
#     fraction = count / n_datapoints**2
#     return fraction


def count_states(H, threshold):
    dist = scipy.spatial.distance_matrix(H, H)
    threshold = 0.2 * np.max(dist)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold
    ).fit(H)
    count = len(set(clustering.labels_))
    n_datapoints = H.shape[0]
    fraction = count / n_datapoints
    return fraction


# def count_states(H, threshold):
#     dist = scipy.spatial.distance_matrix(H, H)
#     fraction = 1 / dist
#     fraction = np.sum(np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0))
#     # fraction = np.sum(np.nan_to_num(1e-100 * fraction, nan=0.0, posinf=1.0, neginf=0.0))
#     return fraction


class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {"train", "val"}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == "train":
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == "val":
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)


def train(config):
    print("using config:", config)
    train_cfg = config["train"]
    wandb_cfg = config["wandb"]
    if wandb_cfg["use_wandb"]:
        wandb.init(project=wandb_cfg["wandb_project"], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_item(config["dataset"])
    train_data = GroupDataset(dataset, "train")
    val_data = GroupDataset(dataset, "val")
    model = load_item(config["model"], dataset.n_vocab, dataset.n_out, device)
    model.train()
    train_dataloader = DataLoader(
        train_data, num_workers=train_cfg["num_workers"], batch_size=train_cfg["bsize"]
    )
    val_dataloader = DataLoader(
        val_data, num_workers=train_cfg["num_workers"], batch_size=train_cfg["bsize"]
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=train_cfg["betas"],
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: min(s / train_cfg["warmup_steps"], 1)
    )
    step = 0
    n_states = 0  # should probably be set to something else
    count_thresh = None
    for x, y in tqdm(train_dataloader):
        loss, logs = model.get_loss(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step()
        if (step) % train_cfg["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs, hiddens = [], []
                for i, (val_x, val_y) in tqdm(enumerate(val_dataloader)):
                    if i >= train_cfg["eval_batches"]:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
                    hidden = model.get_hidden(x.to(device)).detach().cpu().numpy()
                    hiddens.append(hidden)
                H = np.concatenate(hiddens, axis=0)
                if not count_thresh:
                    dist = scipy.spatial.distance_matrix(H, H)
                    count_thresh = 0.1 * np.max(dist)
                n_states = count_states(H, threshold=count_thresh)
                # if not count_thresh:
                #     count_thresh = n_states
                # n_states = n_states / count_thresh
            out_log = {
                "val": combine_logs(all_val_logs),
                "train": combine_logs([logs]),
                "step": (step + 1),
                "lr": float(lr_schedule.get_last_lr()[0]),
                "n_states": n_states,
            }
            print(out_log)
            if wandb_cfg["use_wandb"]:
                wandb.log(out_log)
            model.train()
        step += 1
        if train_cfg["max_steps"] is not None and step >= train_cfg["max_steps"]:
            break


@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
