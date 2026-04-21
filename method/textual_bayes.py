import logging
from dataclasses import replace
import os
import pickle
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import ListConfig
from textgrad.engine import EngineLM

from utils.checkpointing import load_from_disk, save_to_disk

from .models import EnsembleModel, ParamData
from .tgext.utils import standardize_engine

logger = logging.getLogger("TextualBayes")


def textual_bayes_mc(chain_dataloader, eval_dataloader, cfg):
    # Check if we have a saved model to load
    method_cfg = cfg.method
    data_cfg = cfg.data

    transform = instantiate(data_cfg.transform)
    output_dir = Path(cfg.output_dir)

    if "checkpoint_path" in cfg:
        logger.info(f"Loading checkpoint from {cfg.checkpoint_path}")
        ensemble = load_from_disk(cfg.checkpoint_path)
        # out_file = (
        #         f"{cfg.output_dir}/steps_{cfg.method.steps}"
        #         f"_chains_{cfg.method.num_chains}"
        #         f"_burn_in_{cfg.method.burn_in}.pkl"
        #     )
        # if os.path.exists(out_file):
        #     logger.info(f"Found existing model at {out_file}, loading...")
        #     with open(out_file, "rb") as f:
        #         save_dict = pickle.load(f)
        #         if "ensemble" in save_dict:
        #             logger.info("Loading saved ensemble parameters...")
        #             # Create ensemble with saved parameters
        #             method_cfg = cfg.method
        #             data_cfg = cfg.data
        #             init_params_data = [ParamData(**p_data) for p_data in data_cfg.params_data]
        #             ensemble = []
        #             for params_data in save_dict["ensemble"]:
        #                 model = instantiate(method_cfg.model)(init_params_data)
        #                 for param, saved_param in zip(model.parameters(), params_data):
        #                     param.value = saved_param
        #                 ensemble.append(model)
        #             ensemble = EnsembleModel(models=ensemble)
        #             logger.info("Model loaded successfully")
        #         else:
        #             logger.info("No saved ensemble parameters found, will train new model")
        # else:
        #     logger.info("No saved model found, will train new model")

        #     # If we get here, we need to train a new model
        #     logger.info("Training new model...")

    else:
        # Set some params for generating the sample
        steps = method_cfg.steps
        burn_in = method_cfg.burn_in
        thinning = method_cfg.thinning
        num_repeats = method_cfg.num_repeats

        assert isinstance(data_cfg.params_data, ListConfig)
        # Each param is defined as a DictConfig in the config so now we parse into a dataclass
        init_params_data = [ParamData(**p_data) for p_data in data_cfg.params_data]
        if method_cfg.num_chains <= 1:
            chains_init_params_data = [init_params_data]
        else:
            chains_init_params_data = [init_params_data] + reword_params(
                init_params_data,
                method_cfg.num_chains - 1,
                standardize_engine("gpt-4o-mini"),
            )

        # We will populate this list with MCMC chain samples
        # It is a nested list of all samples from all chains for all params
        chains_samples_params_data: list[list[list[ParamData]]] = []

        # Each iteration of this loop represents a single MCMC chain
        for params_data in chains_init_params_data:
            # Create the model, optimizer, proposal distribution, and mcmc method from the config
            model = instantiate(method_cfg.model)(params_data)
            model_params = model.parameters()
            # Check that the model was instantiated correctly
            assert len(model_params) == len(params_data)
            assert all([p.requires_grad for p in model_params])

            optimizer = instantiate(method_cfg.optimizer)(parameters=model_params)
            proposal = instantiate(method_cfg.proposal)(
                model=model,
                optimizer=optimizer,
                prior_losses_text=data_cfg.get("prior_losses_text", None),
            )
            mcmc = instantiate(method_cfg.mcmc)(proposal=proposal)

            # Obtain samples of params from this single chain
            samples_params = mcmc.sample_from_chain(
                chain_dataloader,
                steps,
                burn_in,
                thinning,
                transform=lambda x: transform(x[0]),
                **method_cfg.chain_kwargs,
            )
            # Add the param data fields to each sample from the chain
            samples_params_data = [
                [replace(p_data, value=p.value) for p_data, p in zip(params_data, params)]
                for params in samples_params
            ]
            chains_samples_params_data.append(samples_params_data)

        # Ensemble the samples
        ensemble = []
        for samples_params_data in chains_samples_params_data:
            for params_data in samples_params_data:
                for _ in range(num_repeats):
                    ensemble.append(instantiate(method_cfg.model)(params_data))
        ensemble = EnsembleModel(models=ensemble)

    # Save the ensemble
    output_path = output_dir / "ensemble.pkl"
    logger.info(f"Saving checkpoint to {output_path}")
    save_to_disk(ensemble, output_path)

    logger.info("Final system prompts")
    for model in ensemble.models:
        logger.info(model.system_prompt.value)

    # Evaluate the ensemble
    evaluator = instantiate(data_cfg.evaluator)(
        eval_dataloader,
        ensemble,
        transform=transform,
        output_dir=output_dir,
        **method_cfg.eval_kwargs,
    )
    evaluator.evaluate()


def reword_params(
    params_data: list[ParamData], num_rewordings: int, engine: EngineLM
) -> list[list[ParamData]]:
    def reword_single_param(
        param_data: ParamData, num_rewordings: int, engine: EngineLM
    ) -> list[ParamData]:
        param_rewordings = engine(
            f"Create {num_rewordings} rewordings of the following prompt "
            f"separated by the character | with no spaces: "
            f"{param_data.value}"
        ).split("|")
        param_rewordings = [p.strip() for p in param_rewordings]
        # Add the param data fields to each rewording
        param_rewordings_data = [replace(param_data, value=p) for p in param_rewordings]
        return param_rewordings_data

    params_rewordings_data = list(
        zip(*[reword_single_param(p_data, num_rewordings, engine) for p_data in params_data])
    )
    return params_rewordings_data
