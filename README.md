<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

[![Paper](https://img.shields.io/badge/arXiv-2506.10060-b31b1b?logo=arxiv&logoWidth=10&link=https://arxiv.org/abs/2506.10060)](https://arxiv.org/abs/2506.10060)

# Textual Bayes

This is the codebase for the ICLR 2026 work [*Textual Bayes: Quantifying Prompt Uncertainty in LLM-Based Systems*](https://arxiv.org/abs/2506.10060) accepted to [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009142).

## Installation:
Create a conda environment using:

```
conda create -n bbt python=3.12.0
conda activate bbt
```

Install dependencies
```
pip install -r requirements.txt
```

Download the datasets:
```
cd datasets
bash prepare_datasets.sh
cd ..
```

Make sure to also set up OpenAI and Together AI keys. Then set the following environment variables.
```
export OPENAI_API_KEY="your open ai key"
export TOGETHER_API_KEY="your together ai key"
```

## Run

```
python main.py +data=$DATA +method=$METHOD
```
Options for each variable above:
```
DATA: {mmlu, gsm8k, ...}
METHOD: {textual_bayes, baseline}
```

## Running conformal factuality experiments
```
python main.py +data=factscore +method=textual_bayes_factuality_factscore
```

## Development
Please run the following before merging a PR:
```
# Code formatting:
black -l 100 .

# Import sorting:
isort . --line-length 100
```

You can run tests with `pytest`:
```
pytest -s
```

# Citation
If you find our work helpful in some way, please consider citing us in yours:
```
@inproceedings{ross2026textual,
    title={Textual Bayes: Quantifying Prompt Uncertainty in {LLM}-Based Systems},
    author={Brendan Leigh Ross and No{\"e}l Vouitsis and Atiyeh Ashari Ghomi and Rasa Hosseinzadeh and Ji Xin and Zhaoyan Liu and Yi Sui and Shiyi Hou and Kin Kwan Leung and Gabriel Loaiza-Ganem and Jesse C. Cresswell},
    booktitle={International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=VPmsAr1OTl}
}
```
# License

This data and code is licensed under the MIT License, copyright by Layer 6 AI.
