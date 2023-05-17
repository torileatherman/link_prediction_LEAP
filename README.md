# Link Prediction using Learnable Topology Augmentation (LEAP)

Pytorch implementation of the method LEAP, proposed by Ahmed E. Samy, and explored in my master thesis. This method, LEAP, can be used to perform link prediction on graph-type data, under both inductive and transductive settings.

## Setup
Download or clone the repository and open in code editor. 

### Create environment and Install Packages
If using Anaconda or Miniconda:
  ``` 
  $ conda create --name <env_name> --file requirements.txt
  ```

If using pip,
  ```
  $ pip install -r requirements.txt
  ```

## Usage
Data splits from the datasets [Wikipedia (Chameleon) and Wikipedia (Crocodile)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikipediaNetwork.html#torch_geometric.datasets.WikipediaNetwork), [PubMed](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid), [Twitch](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Twitch.html#torch_geometric.datasets.Twitch) are preprocessed and split into training, validation, and testing sets, ready for use. 

### Files
[utils.py](utils.py) contains the hyperparameters used in our model, LEAP. The parameters that are useful to modify by the user are as follows:
1. Transductive: 
    If set to ``default= TRUE``, LEAP will operate under transductive settings.
    If set to ``default= FALSE``, LEAP will operate under inductive settings.
2. Device: 
    The default for device is set to operate on a CUDA device. If one is not available, set ``default= 'CPU'``.
3. Name: The name of the dataset LEAP should be used on. The options include: 'Wikipedia', 'crocodile', 'PubMed', 'Twitch'. Note that these names are case sensitive.
4. Epochs: The number of epochs used for training LEAP; can take any integer value. 

[leap.py](leap.py) contains the code for implementation of the LEAP method. It imports classes from the files [MLP.py](https://github.com/torileatherman/link_prediction_LEAP/tree/main/models/MLP.py) and [model.py](https://github.com/torileatherman/link_prediction_LEAP/tree/main/models/model.py) for use. This will train and validate the model through the predetermined epochs, and output a testing score in terms of AUC and AP.

### To run
After setting the desired hyperparameters in [utils.py](utils.py), run the method LEAP using the following in the terminal:
```
$ python leap.py
```









