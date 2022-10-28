# RNAcompete_classifier
This repository contains the code and data for training the RNAcompete experiment classifier used in Ray *et al.* 2022. To reproduce the results in the paper, follow the instructions below.

## File Descriptions

`RNAcompete_LR_classifier.py`: Contains the Python script that trains the classifier, tests the classifier, and applies the classifier to all ucRBP experiments.

`training_set.txt`: Tab-delimited file containing the feature matrix for training. Each line is an individual RNAcompete experiment. The first column (class) indicates whether it was a passed (1) or failed (0) experiment. Subsequent columns contain values for each of the features as described in the paper.

`testing_set.txt`: Tab-delimited file with the same format as `training_set.txt`. This file contains the 40 experiments used to test the classifier.

`ucrbp_experiment_features.txt`: Tab-delimited file with the same format as `training_set.txt` and `testing_set.txt`, EXCEPT for the 'class' column. Contains feature values for all ucRBP experiments (full-length ucRBPs and truncated constructs). 

`train_IDs.txt`: Contains the RNAcompete experiment identifiers for all experiments in `training_set.txt`.

`test_IDs.txt`: Contains the RNAcompete experiment identifiers for all experiments in `testing_set.txt`.

`ucrbp_IDs.txt`: Contains the RNAcompete experiment identifiers for all experiments in `ucrbp_experiment_features.txt`. IDs in this file correspond to IDs in Supplementary Table 1.

For all `*IDs.txt` files, the IDs correspond line-by-line to the values in `*_set.txt` and `ucrbp_experiment_features.txt`. For example, to directly view the feature values for ucRBP experiments by ID, the files can be merged as follows:

  `paste ucrbp_IDs.txt ucrbp_experiment_features.txt`

## Train & Apply the Classifier

Clone the repository and ensure the following requirements are installed:

- python3 version 3.8
- numpy version 1.19.2
- sklearn version 0.23.2
- skopt version 0.8.1
- pickleshare version 0.7.5

Run the following from within the directory:

  `python RNAcompete_LR_classifier.py`

## Output File Descriptions

`LR_model.sav`: Contains a "pickled" version of the logistic regression model trained in `RNAcompete_LR_classifier.py`. This model can be loaded in python by running the following:
  ```
  import pickle
  lr = pickle.load(open('LR_model.sav', 'rb'))
  ```

`LR_coefficients.txt`: Tab-delimited file containing one line for each feature along with the weight of the feature in the model.

`test_set_AUROC.txt`: Contains the AUROC for the test set.

`test_set_probability_estimates.txt`: Contains one line per test set experiment with the probability estimate that the experiment belongs to class 1 (passed experiments).

`ucrbp_probability_estimates.txt`: Contains one line per ucRBP experiment with the probability estimate that the experiment belongs to class 1 (passed experiments).

For both `*_probability_estimates.txt` files, the probability estimates correspond line-by-line to the associated `*IDs.txt` files (and thus the files containing the features as well). For example, to directly view the probability estimates for ucRBP experiments by ID, the files can be merged as follows:

  `paste ucrbp_IDs.txt ucrbp_probability_estimates.txt`


  
