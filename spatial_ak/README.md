# Wildfire Mapping with Attentive Kernel

## Installation

```
conda create -n wildfire python=3.8
conda activate wildfire  
pip install -r requirements.txt
pip install -e .
```

## Get Started

```bash
cd experiments
python main.py kernel=rbf
python main.py kernel=ak
```
