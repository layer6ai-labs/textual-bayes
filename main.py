import logging
from datetime import datetime

import hydra
import torch
from dotenv import load_dotenv
from namesgenerator import get_random_name
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from method.baseline.runner import run_baseline
from method.textual_bayes import textual_bayes_mc
from utils.data import dataset_classes

OUTPUT_DIR = None


def get_output_dir():
    global OUTPUT_DIR
    if OUTPUT_DIR is None:
        now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        OUTPUT_DIR = f"outputs/{now_time}_{get_random_name()}/"
    return OUTPUT_DIR


OmegaConf.register_new_resolver("get_output_dir", get_output_dir)


@hydra.main(version_base=None, config_path="conf/", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    method_cfg = cfg.method
    log = logging.getLogger("Main")
    log.info(OmegaConf.to_yaml(cfg))

    # Set seed
    generator = torch.Generator().manual_seed(cfg.random_seed)

    # Load environment variables
    load_dotenv()

    dataset = cfg.data.dataset
    log.info(f"\n{'=' * 100}\nDataset: {dataset}\n")
    chain_dataset = dataset_classes[dataset](
        split=data_cfg.chain_split,
    )
    eval_dataset = dataset_classes[dataset](
        split=data_cfg.eval_split,
    )

    chain_dataloader = DataLoader(
        chain_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        generator=generator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    if method_cfg.method == "textual-bayes":
        result = textual_bayes_mc(chain_dataloader, eval_dataloader, cfg)
    elif method_cfg.method == "baseline":
        log.info("Running Baseline method...")
        run_baseline(eval_dataloader, cfg)
        log.info("Baseline method finished.")

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
