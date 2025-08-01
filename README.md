# U.S. television news coverage of climate change policy is aggregately balanced but polarized

This repository houses the code for the research analyzing climate policy coverage in U.S. television news.

## Installation instructions
Use `git clone` to download this repository either via HTTPS

```shell
git clone https://github.com/kathlandgren/policy-misperception.git
```

or SSH

```shell
git clone git@github.com:kathlandgren/policy-misperception.git
```

### Create Python virtual environment

Create a Python virtual environment in the root of the publishorcomparish directory:

  ```shell
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
## Data availability

- Transcript annotation data are available for open access at [OSF](https://osf.io/r6sf5/?view_only=c2804dd7c80a467885df9eccdd781e3b).
- Transcript text data are available by subscription from the Nexis-Uni database, a public library research platform and a subsidiary of LexisNexis and the RELX Group.
- Survey data used in this work are available through the following publication: 
Sparkman, Gregg, et al. "Americans experience a false social reality by underestimating popular climate policy support by nearly half." *Nature Communications* 13.1 (2022): 4779. [DOI:10.1038/s41467-022-32412-y](https://doi.org/10.1038/s41467-022-32412-y)

## Contents

The repository is organized into the following directories:

- [00_data_processing/](https://github.com/kathlandgren/policy-misperception/tree/main/00_data_processing): Includes Jupyter notebooks for processing and formatting annotation data for general climate policy analysis, as well as climate policy analysis by category.
- [01_intercoder_reliability/](https://github.com/kathlandgren/policy-misperception/tree/main/00_data_processing): Includes Jupyter notebooks and Python scripts for calculating intercoder reliability for both rounds of annotation (general and by category).
- [02_fig_2_linking_to_survey_data/](https://github.com/kathlandgren/policy-misperception/tree/main/02_fig_2_linking_to_survey_data): Contains Jupyter notebooks for linking the processed annotation data to survey data and computing synthetic ratings, as well as generating Fugure 2.
- [03_fig_1_policy_valence/](https://github.com/kathlandgren/policy-misperception/tree/main/03_fig_1_policy_valence): Contains Jupyter notebooks for generating Figure 1, which shows the distribution of transcripts expressing support, opposition, or neutrality towards climate policy, as well as the differences between media outlets. Additionally plots the weighted estimates based on the survey data.
- [04_fig_3_policy_category_analysis/](https://github.com/kathlandgren/policy-misperception/tree/main/04_fig_3_policy_category_analysis): Contains Jupyter notebooks for generating Figure 3, which shows the distribution of attitudes toward different climate policy categories, as well as the differences between media outlets, as well as the prevalence of different policy categories in the corpus.
- [05_SI_robustness/](https://github.com/kathlandgren/policy-misperception/tree/main/05_SI_robustness): Contains Jupyter notebooks for generating the supplementary information figures, which include robustness checks and additional analyses of the data.