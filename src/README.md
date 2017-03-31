# Exploration of the MNIST dataset

## Code Folder

The code directory contains all codes.

### run.py
Run the model and Save the results (Raw results and Graphs) from the console.
It will load or download the dataset, train the model and predict the results.
example: 
``` python run.py -i 10 -l 20 -f output_0 
```
arguments:
- Level of noise in the images, the noise is a random normal variable, you can choose the standard deviation by using ( -i 20 ), the value should be between 0 and 255, default is 0
- Level of randomness in the labels, randomly select and randomly replace a percentage of the labels, you can choose this percentage by using ( -l 40 ), the value should be between 0 and 100, default is 0
- Folder where to save the figures and raw results, this folder will be inside output/figs and output/raw, you can choose it by doing ( -f output_0 ), default is default

### main.py
The main.py file will run the complete analysis I've made by recreating all the figures for the 3 parts of the project.

### Installation

I'm using keras, with theano as a backend.

Other Packages used:
- pandas
- numpy
- matplotlib
- seaborn
