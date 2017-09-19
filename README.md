# Prerequisites

[Anaconda Distribution for Python 3.6](https://www.anaconda.com/download/)

# Installation

Create a new virtual environment through Conda with the specifications listed in the environment.yml file. You can replace myenv with your desired name for the environment.

```
conda env create -f environment.yaml -n myenv
```

Activate the virtual environment

```
source activate myenv
```

In order to work within the virtual environment through Jupyter, we must create an IPython kernel to work within. 

```
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"

```

Be sure to change the kernel in the Jupyter Notebook to our new IPython kernel.

# dots-reversal-ideal-obs
Python code that models ideal-observer for dots-reversal task

Consult [Wiki pages](https://github.com/aernesto/dots-reversal-ideal-obs/wiki) for description of code functionalities
