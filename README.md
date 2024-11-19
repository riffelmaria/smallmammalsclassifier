# Python repository for a small terrestrial mammals classifier
Python repository for training, prediction, and evaluation of a small terrestrial mammal classifier

---

## Contents
- [Python repository for a small terrestrial mammals classifier](#python-repository-for-a-small-terrestrial-mammals-classifier)
  - [Contents](#contents)
  - [Description](#description)
  - [Components](#components)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dataset](#dataset)
  - [Models](#models)
  - [Contributing](#contributing)
  - [License](#license)

---

## Description

This Python project contains all steps to build a machine-learning classifier designed to recognize and classify small terrestrial mammal calls, including species groups like rodents, voles, and shrews. Also, scripts for prediction and evaluation are available, i.e. using the provided models.
Built with Python, this tool can be used by researchers and enthusiasts for detecting calls in Passive Acoustic Monitoring (PAM) audio files, understanding habitat diversity, and aiding conservation efforts.

## Components

- **Species Detection**: Identify calls of various small terrestrial mammals in audio files.
- **Pre-trained Model**: Use a pre-trained model to get started immediately or train your own model.
- **Audio Preprocessing**: Automatically creates audio chunks from long audio files for easier postprocessing.
- **Easy Integration**: Simple to use with any Python environment and customizable for specific use cases.

## Installation

To get started with the Small Terrestrial Mammals Classifier, clone the repository and install the required dependencies.

Execute these lines on a Linux OS:

```bash
# Make sure to have relevant packages installed
$ sudo apt-get install git  && sudo apt-get install python3-venv

# Clone the repository to your desired directory
$ cd /path/to/projects && git clone https://github.com/riffelmaria/smallmammalsclassifier.git

# Change directory to the project
$ cd smallmammalsclassifier

# Pull repository
$ git pull

# Install a Python venv
$ python3.11 -m venv /path/to/.venv

# Activate the venv
$ source /path/to/.venv/bin/activate

# Adjusting local variables ...?

# Install Python project ...
$ pip install -e .

# Now, all functions are ready to use!
```

And these lines on a Windows OS:

```bash
# Make sure to have relevant packages installed
$ winget install --id Git.Git -e --source winget
$ winget install -e --id Python.Python.3.11 --scope machine

# Clone the repository to your desired directory
$ cd C:\path\to\projects && git clone https://github.com/riffelmaria/smallmammalsclassifier.git

# Change directory to the project
$ cd smallmammalsclassifier

# Make sure to have the virtual environment (venv) installed
$ pip install virtualenv    

# Create a new Python venv
$ py -m venv C:\path\to\new\virtual\environment\myvenv

# Activate the venv in PowerShell
$ C:\path\to\new\virtual\environment\myvenv\Scripts\Activate.ps1

# Alternatively, activate the venv in cmd.exe
$ C:\path\to\new\virtual\environment\myvenv\Scripts\activate.bat

# Adjusting local variables ...?

# Install Python project ...
$ pip install -e .

# Now, all functions are ready to use!
```

## Usage

After installing, you can use all the functions in the project and the classifier on your data. 
Here's a quick example of how to get started with a CLI:

```bash
# Preprocess data for prediction
$ ....

# Choose a model and predict on small sample of your annotated data
$ ...

# Choose a custom threshold if needed and start evaluation
$ ...

# Predict all data
$ ...

```

## Dataset

The dataset used for training the provided models includes audio data of various small terrestrial mammals. The dataset itself is not included in this repository for storage and licensing reasons. However, you can find available data under [BTO (British Trust of Ornithology)](https://www.bto.org/our-science/publications/peer-reviewed-papers/acoustic-identification-small-terrestrial-mammals "The acoustic identification of small terrestrial mammals in Britain") and [ChiroVox](https://www.chirovox.org/ "The bat call library").
Additionally, if you need the data I annotated with a coarse class scale (small terrestrial mammals, noise, and bats), please contact me at git@mariariffel.de or the Chair of Geobotany at the University of Freiburg, Germany.

Please make sure to follow the annotation scheme as in ./src/shared/groups_minimal.py or ./src/shared/genera.py, or simply add your own annotations to match the class scales.

To train the model on your own dataset, please confirm that your audio and Raven data can be found in the same folder, like so:

```
dataset/
├── species_1/
│   ├── audio1.WAV
│   ├── audio1.selections.Table.1.txt
│   ├── audio2.WAV
│   ├── audio2.selections.Table.1.txt
├── species_2/
│   ├── audio1.WAV
│   ├── audio1.selections.Table.1.txt
│   ├── audio2.WAV
│   ├── audio2.selections.Table.1.txt
...
```

## Models

The models provided are built using the OpenSoundscape library (internally uses PyTorch). They are trained on an audio dataset of small terrestrial mammal calls.
The model X achieves approximately X% target recall and Z% target ... in identifying the species groups .... 
The model Y achieves approxiamtely Y% target recall and Z% ...
You can find the training script in train.py.

If you want to train your own model, please adjust the parameter .. with this script ... before executing train.py...

## Contributing

Contributions are welcome :)
If you have ideas for improving the code, the classifier, adding new species, or enhancing recall, precision, or F1, please submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
