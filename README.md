# Why does the U.S. public underestimate climate policy support?

This repository houses the code for the research on the mechanisms behind public support on climate policy.

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
Sparkman, Gregg, Nathan Geiger, and Elke U. Weber. "Americans experience a false social reality by underestimating popular climate policy support by nearly half." *Nature Communications* 13.1 (2022): 4779. [DOI:10.1038/s41467-022-32412-y](https://doi.org/10.1038/s41467-022-32412-y)

## Contents

The repository is organized into the following directories:
- [network_simulations/](https://github.com/kathlandgren/policy-misperception/tree/main/network_simulations): Contains the code related to the network simulations and analyses presented in the manuscript. The initial homophily analysis is based on the following: Karimi, Fariba, et al. "Homophily influences ranking of minorities in social networks." *Scientific Reports* 8.1 (2018): 11077. [DOI:10.1038/s41598-018-29405-7](https://doi.org/10.1038/s41598-018-29405-7). The analysis is divided into the following sections:
  - opinion and perception functions
  - homophily only analysis
  - homophily with promotion of minority viewpoint analysis
- [transcript_analysis/](https://github.com/kathlandgren/policy-misperception/tree/main/transcript_analysis): Includes scripts pertinent to the transcript annotation process. This folder includes the code used for assessing intercoder reliability as well as the code used for generating the estimated viewership for each of the news outlets analyzed in the manuscript.
