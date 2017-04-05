## Organisation

1. In the notebooks folder, the `data explo.ipynb` shows the thought process behind the features
2. At the root, the `manage.py` entry points allows to train, predict & optimize hyper parameters
3. code is in the `credit` folder

## setup

In case of library version / import problems.

1. install miniconda

https://conda.io/miniconda.html

2. create virtual env

at repo's root (where `environment.yml` is present)

`conda-env create`

## Usage

first activate the virt-env

`source activate kagcredit`

1. train

Perfoms CV, then trains <algo> on entire data set and saves the model.

`python manage.py train <algo>`

2. hyper params optimization

Need to define the optimization grid in HYPER_OPTI global in manage.py

`python manage.py train <algo> --hyper`

2. predict

Creates a prediction file with <algo>, with the latest saved model for <algo>

`python manage.py predict <algo>`


