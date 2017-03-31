# Exploration of the MNIST dataset

## Figs Folder

The figs directory contains graphs and images.

### Part 1: CNN trained on the original MNIST dataset
- output_0

### Part 2: Gaussian Noise added to the Images
- output_1: std=8
- output_2: std=32
- output_3: std=128

### Part 3: Fixed Percentage of the Labels randomized
- output_4: percentage = 5%
- output_5: percentage = 15%
- output_6: percentage = 50%

### Summary of the findings

|Output|Training Acc|Training Acc|
|--------|-----|-----|
|Output_0|95.32|98.59|
|Output_1|95.02|98.65|
|Output_2|94.90|98.68|
|Output_3|88.70|97.99|
|Output_4|90.87|98.51|
|Output_5|82.05|98.39|
|Output_6|51.37|97.62|
