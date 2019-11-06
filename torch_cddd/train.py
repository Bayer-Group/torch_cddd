import argparse
import numpy as np
import os
import torch
import ray
from ray import tune
from ray.tune import Trainable, run, schedulers
from torch_cddd.models import CDDDSeq2Seq
from torch_cddd.data import SmilesTokenizer,  SmilesDataset, SmilesDataLoader, TOKENS, make_data_loader, batch_to_device
from torch_cddd.hyperparameters import add_arguments
from torch_cddd.evaluate import evaluate_qsar, evaluate_reconstruction
from torch.nn.utils import clip_grad_norm_

TRAIN_PATH = "~/projects/torch_cddd/data/train.csv"
#TRAIN_PATH = "~/projects/torch_cddd/data/eval.csv"
EVAL_PATH = "~/projects/torch_cddd/data/eval.csv"
AMES_PATH = "~/projects/torch_cddd/data/ames.csv"
LIPO_PATH = "~/projects/torch_cddd/data/lipo.csv"
BATCH_SIZE = 64
NUM_WORKERS = 4
EVAL_FREQ = 2000

CONFIG = {
    "emb_size": tune.grid_search([512]),
    "rnn_hidden_size": tune.grid_search([1024, 2048]),
    "rnn_num_layers": tune.grid_search([3, 4]),
    "lr": tune.grid_search([0.0001, 0.0002]),
    "start_tf_rate": tune.grid_search([1]),
    "tf_step_freq": tune.grid_search([10, 50]),
    "tf_step_rate": tune.grid_search([1.0]),
    "emb_noise_std": tune.grid_search([0.001, 0.05]),
    "input_dropout": tune.grid_search([0.0, 0.2]),
    "input_noise_std": tune.grid_search([0.0]),
}

#TODO: include lstm and bidirectional


class TrainCDDD(Trainable):
    def _setup(self, config):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_iterator = make_data_loader(
            csv=TRAIN_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            sampler="infinit",
        )
        self.eval_loader = make_data_loader(
            csv=EVAL_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            drop_last=True
        )
        self.qsar_loader_ames = make_data_loader(
            csv=AMES_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            label_col_header="label",
            rdkit_descriptors=False,
            input_type="canonical"
        )
        self.qsar_loader_lipo = make_data_loader(
            csv=LIPO_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            label_col_header="label",
            rdkit_descriptors=False,
            input_type="canonical"
        )
        self.train_iterator = iter(self.train_iterator)
        self.model = CDDDSeq2Seq(
            rnn_hidden_size=config["rnn_hidden_size"],
            rnn_num_layers=config["rnn_num_layers"],
            emb_size=config["emb_size"],
            token_emb_size=32,
            predictor_hidden_size=[128, 32],
            num_properties=7,
            input_noise_std=config["input_noise_std"],
            emb_noise_std=config["emb_noise_std"],
            input_dropout=config["input_dropout"]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config["lr"]
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.5, patience=20, cooldown=50
        )
        self.eval_freq = EVAL_FREQ
        self.tf_ratio = config["start_tf_rate"]
        self.tf_step_freq = config["tf_step_freq"]
        self.tf_step_rate = config["tf_step_rate"]

    def _maybe_update(self, eval_acc):
        if (self.iteration + 1) % self.tf_step_freq == 0:
            self.tf_ratio *= self.tf_step_rate
        self.lr_scheduler.step(eval_acc)
        self.lr = self.optimizer.param_groups[0]["lr"]

    def _train(self):
        train_loss = train(
            model=self.model,
            optimizer=self.optimizer,
            dataloader=self.train_iterator,
            tf_ratio=self.tf_ratio,
            device=self.device,
            num_steps=self.eval_freq
        )
        eval_loss, eval_acc = evaluate_reconstruction(
            model=self.model,
            dataloader=self.eval_loader,
            device=self.device
        )
        ames_score = evaluate_qsar(
            model=self.model,
            dataloader=self.qsar_loader_ames,
            device=self.device,
            clf_type="SVC"
        )
        lipo_score = evaluate_qsar(
            model=self.model,
            dataloader=self.qsar_loader_lipo,
            device=self.device,
            clf_type="SVR"
        )
        self._maybe_update(eval_acc)
        log_dict = {
            "train_loss": train_loss,
            "eval_accuracy": eval_acc,
            "eval_loss": eval_loss,
            "ames_score": ames_score,
            "lipo_score": lipo_score,
            "tf_ratio": self.tf_ratio,
            "lr": self.lr,
        }
        return log_dict

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "tf_ratio": self.tf_ratio
        }, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tf_ratio = checkpoint['tf_ratio']


def train(model, optimizer, dataloader, tf_ratio, device, num_steps):
    model.train()
    train_loss = 0
    for step in range(num_steps):
        batch = next(dataloader)
        train_loss += train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            teacher_forcing_ratio=tf_ratio,
            device=device
        )
    train_loss /= num_steps
    return train_loss


def train_step(model, optimizer, batch, teacher_forcing_ratio, device):
    input_tensor, input_length, target_tensor, labels = batch_to_device(batch, device)
    loss, out = model.forward(input_tensor, input_length, target_tensor, labels, teacher_forcing_ratio)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", help="Address of ray cluster", default="auto")
    parser.add_argument("--ray-run-local-dir", help="Local dir for ray.tune.run", default=None)
    parser = add_arguments(parser)
    flags_, _ = parser.parse_known_args()
    async_hb_scheduler = schedulers.AsyncHyperBandScheduler(
        time_attr='training_iteration',
        metric='lipo_score',
        mode='max',
        max_t=600,
        grace_period=15,
        reduction_factor=3,
        brackets=3)
    ray.init(
        address=flags_.ray_address,
    )
    run(
        run_or_experiment=TrainCDDD,
        name="hyperband5",
        resources_per_trial={
            "cpu": 10,
            "gpu": 0.5},
        scheduler=async_hb_scheduler,
        config=CONFIG,
        local_dir=flags_.ray_run_local_dir,
        reuse_actors=True,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        keep_checkpoints_num=1,
        checkpoint_score_attr='training_iteration'
        )
