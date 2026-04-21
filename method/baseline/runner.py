import json
import logging
import os
from datetime import datetime

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


def run_baseline(eval_dataloader: DataLoader, cfg: DictConfig) -> dict:
    method_cfg = cfg.method
    data_cfg = cfg.data

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("Starting baseline run...")

    shared_n = method_cfg.n
    log.info(f"Perturbation count n = {shared_n}")

    prompt_param_config = data_cfg.params_data[0]
    system_prompt = prompt_param_config.value
    prompt_addendum = prompt_param_config.get("addendum", "")

    system_prompt = system_prompt + prompt_addendum
    log.info(f"Using System Prompt :\n{system_prompt}")

    engine = hydra.utils.instantiate(method_cfg.engine)
    log.info(
        f"Instantiated engine: {type(engine).__name__} with model: {method_cfg.engine.model_name}"
    )

    perturber_key = method_cfg.get("perturber_select")
    perturber_cfg = method_cfg.perturbers[perturber_key]
    perturber = hydra.utils.instantiate(perturber_cfg)
    log.info(f"Instantiated perturber: {perturber_key} ({perturber.__class__.__name__})")
    log.info(f"Perturber config: {perturber_cfg}")
    aggregator_key = method_cfg.get("aggregator_select")
    agg_cfg = method_cfg.aggregators[aggregator_key]
    init_kwargs = {}
    target_str = agg_cfg.get("_target_", "")
    if target_str:
        agg_target_class_name = target_str.split(".")[-1]
        if agg_target_class_name in ["Ask4ConfNumUQ", "Ask4ConfWordUQ"]:
            init_kwargs["engine"] = engine
        elif agg_target_class_name in ["FrequencyUQ", "SemanticFrequencyUQ"]:
            init_kwargs["answer_regex"] = data_cfg.get("answer_regex")
        selected_aggregator = hydra.utils.instantiate(agg_cfg, **init_kwargs)
        log.info(
            f"Instantiated selected aggregator: {aggregator_key} ({selected_aggregator.__class__.__name__})"
        )
    else:
        raise Exception(
            f"Selected aggregator '{aggregator_key}' has no '_target_'. Skipping aggregation."
        )

    if not data_cfg.get("transform"):
        raise ValueError("Transform configuration missing.")
    transform_fn = hydra.utils.instantiate(data_cfg.transform)
    log.info(f"Instantiated data transform function: {data_cfg.transform.get('_target_')}")

    results = []
    num_examples_to_run = method_cfg.eval_kwargs.get("num_examples", -1)
    processed_count = 0
    total_examples = len(eval_dataloader.dataset)
    limit = (
        total_examples if num_examples_to_run == -1 else min(num_examples_to_run, total_examples)
    )
    log.info(f"Will process up to {limit} examples...")

    for batch in tqdm(
        eval_dataloader,
        total=limit if limit <= len(eval_dataloader) else len(eval_dataloader),
        desc="Processing Examples",
    ):
        for example in batch:
            if processed_count >= limit:
                break
            example_id = f"item_{processed_count}"
            processed_input, ground_truth_value = transform_fn(example)
            if hasattr(processed_input, "value"):
                processed_input = processed_input.value
            if hasattr(ground_truth_value, "value"):
                ground_truth_value = ground_truth_value.value
            initial_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": processed_input},
            ]
            perturbed_data_list = perturber.perturb_response(
                n=shared_n, engine=engine, messages=initial_messages
            )
            log.info(f"Got {len(perturbed_data_list)} perturbed responses for {example_id}")

            uncertainty_score = None
            agg_name = selected_aggregator.__class__.__name__
            log.info(f"Running aggregator '{agg_name}' for example {example_id}...")
            aggregate_kwargs = {}
            if agg_name in ["SemanticFrequencyUQ"]:
                aggregate_kwargs = {"question": example["question"], "y_true": ground_truth_value}
            uncertainty_score = selected_aggregator.quantify_uncertainty(
                perturbed_data_list, **aggregate_kwargs
            )
            log.info(f"Result for '{agg_name}': {uncertainty_score}")

            results.append(
                {
                    "raw_input_example": example,
                    "ground_truth": ground_truth_value,
                    "system_prompt": system_prompt,
                    "perturber": perturber_key,
                    "aggregator": aggregator_key,
                    "responses": [x.response for x in perturbed_data_list],
                    # Can log log probs and full perturbations if needed
                    **uncertainty_score,
                }
            )
            if "unanswerable" in example.keys():
                results[-1]["abst_true"] = int(example["unanswerable"])

            processed_count += 1

        if processed_count >= limit:
            log.info(f"Reached processing limit of {limit} examples.")
            break

    instantiate(data_cfg.baseline_eval_fn)(results)

    log.info(f"Finished baseline run. Processed {processed_count} examples.")

    output_dir = cfg.output_dir

    filename = f"{run_timestamp}_{perturber_key}_{aggregator_key}_results.json"
    results_path = os.path.join(output_dir, filename)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")  # Log the new path

    return results
