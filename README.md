# Code for ICLR submission *One for All: Towards Training One Graph Model for All Classification Tasks*

## Requirements
To install requirement for the project using conda:

```
conda env create -f environment.yml
```

## E2E experiments
For joint end-to-end experiments on all collected dataset, run

```
python run_cdm.py num_layers 7 batch_size 512 dropout 0.15 JK none
```

Users can modify the e2e_data_list variable to control which datasets are included during training. The length of e2e_data_list, data_multiple, min_ratio should be the same. They can be specified in command line arguments by comma separated values.

e.g.
```
python run_cdm.py data_list coralink,arxiv d_multiple 1,1 d_min_ratio 1,1
```

OFA-ind can be specified by 

```
python run_cdm.py data_list coralink d_multiple 1 d_min_ratio 1
```

## Low resource experiments
To run the few-shot and zero-shot experiments

```
python run_fs.py batch_size 30 num_epochs 30
```
