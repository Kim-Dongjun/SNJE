# Posterior_Aided_Regularization_for_Likelihood_Free_Inference

Code for reproducing the experiments in the paper submitted to ICML

>"Posterior-Aided Regularization for Likelihood-Free Inference"

## Dependencies

```
python: 3.8
pyTorch: 1.7.1
pyknos 0.14.0
```

## How to reproduce performances

```
The following is the command for the experiments.

python main.py --likelidhoodLearning True --posteriorLearning True --simulation type_your_simulator --thetaDim type_corresponding_input_dimension --xDim type_corresponding_output_dimension --numModes type_corresponding_number_of_modes --simulation_budget_per_round 100 --numRound 30 --device cuda:0

For example, to reproduce the experimental result of SLCP-16, run

python main.py --likelihoodLearning True --posteriorLearning True --simulation SLCP-16 --thetaDim 5 --xDim 50 --numModes 16 --simulation_budget_per_round 100 --numRound 30 --device cuda:0

```