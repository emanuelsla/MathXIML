# Are More Counterexamples Better? 

Explanatory Interactive Machine Learning (XIML) lets machine learning (ML)
and domain experts as well as end-users interact with the decision-making
mechanism of ML models.
Model-agnostic XIML approaches induce the human knowledge
via counterexamples. Some existing model-agnostic XIML procedures
put forward that more counterexamples yield better results.
The purpose of our paper is a formal inspection of this assumption.

---
**Full reference:**

Slany (2024): Are More Counterexamples Better?
Counterexamples and their Effects
in Explanatory Interactive Machine Learning.

---

Supplementary to our formal contribution, we provide experimental evidence
by this repository. Our experimental findings are in line with the
mathematical insights: More counterexamples are not always better;
not for the predictive nor for the explanatory performance of ML models.

## Getting started

Execute the following code cell in your shell to install the requirements
and execute the main script with its default parameters.

```
$ pip install requirements
$ python3 -m scripts.tabbincaipi_experiments
```

Information on the flags can be retrieved by:

```
$ python3 -m scripts.tabbincaipi_experiments -h
```

## File Information

### Overview

```
├── data
│   ├── adult.csv
│   ├── ...
├── modules
│   ├── counterfactual_explainers
│   │   └── apply_dice.py
│   ├── preprocessors
│   │   ├── preprocess_adult_data.py
│   │   ├── ...
│   ├── ui
│   │   ├── Mockup_StreamlitCaipiUI_GermanCreditRisk.py
│   └── utils.py
├── notebooks
│   ├── Benchmark.ipynb
│   ├── CheckPreprocessorsAndLabelers.ipynb
│   └── EvaluateExperiments.ipynb
├── results
│   ├── adult_c0.csv
│   ├── ...
└── scripts
    └── tabbincaipi_experiments.py
```

### Pre-processing and Quality Assessment

The ````data/```` directory contains all data sets as csv.
Their pre-processors and label modifiers can be found in ```modules/preprocessors/```.
The individual files are named accordingly.
The pre-preprocessors and label modifiers are evaluated in
````notebooks/CheckPreprocessorsAndLabelers.ipynb````,
where finally ````notebooks/Benchmark.ipynb```` conducts a benchmark test.

### Experiments and Results

The experiments are implemented in ````scripts/tabbincaipi_experiments.py````,
which utilizes a counterfactual explainer
implemented in ````modules/counterfactual_explainers/apply_dice.py````.
The experimental output is saved to ````results/````.
The findings are aggregated to the tables and figures of the paper
in ````notebooks/EvaluateExperiments.ipynb````.

### User Interface

The supplementary user interface is implemented in
``modules/ùi/Mockup_StreamlitCaipiUI_GermanCreditRisk.py``.
It involves end-users into the optimization process.
