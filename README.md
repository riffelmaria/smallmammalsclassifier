# Python repository for a small terrestrial mammals classifier
Python repository for training, prediction, and evaluation of a small terrestrial mammal classifier

---

## Contents
- [Python repository for a small terrestrial mammals classifier](#python-repository-for-a-small-terrestrial-mammals-classifier)
  - [Contents](#contents)
  - [Description](#description)
  - [Components](#components)
  - [Installation](#installation)
  - [Usage and recommended workflow](#usage-and-recommended-workflow)
    - [Split audio data into clips for prediction](#split-audio-data-into-clips-for-prediction)
    - [Classification with a pre-trained model](#classification-with-a-pre-trained-model)
      - [Classify a test dataset ...](#classify-a-test-dataset-)
      - [... and evaluate the results](#-and-evaluate-the-results)
      - [Classify all audio data](#classify-all-audio-data)
    - [Train a classifier](#train-a-classifier)
      - [Prepare training files](#prepare-training-files)
      - [Training parameters](#training-parameters)
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
$ sudo apt-get install git  && sudo apt-get install python3.11 && sudo apt-get install python3-venv

# Clone the repository to your desired directory
$ cd /path/to/projects
$ git clone https://github.com/riffelmaria/smallmammalsclassifier.git

# Change the directory to the project
$ cd smallmammalsclassifier

# Pull repository
$ git pull

# Create a Python venv
$ python3.11 -m venv /path/to/.venv

# Activate the venv
$ source /path/to/.venv/bin/activate

# Install Python project ...
$ pip install -e .

# Now, all functions are ready to use!
```


For **Windows OS**, you may follow these instructions:

1. Install git on your computer: Follow the instructions on this [website](https://git-scm.com/downloads/win "Git Download for Windows") or this [video](https://www.youtube.com/watch?v=7jPdEtsTSIE "How to install and configure git with GitHub in Windows 11").
2. You may install the GitHub Desktop App from this [website](https://desktop.github.com/download/ "Download GitHub Desktop") and sign in the app with your GitHub account.
3. Clone this repository on your computer, either with the app or with this line in a project folder PYPROJ
    ```bash
    cd C:\Users\NAME\PYPROJ
    git clone https://github.com/riffelmaria/smallmammalsclassifier.git
    ```
4. Make sure the **correct Python Version 3.11** is installed.
   1. For that, **first** make sure that no other Python version is installed: In a PowerShell, cmd.exe or IDE type
    ```
    python --version
    ```
   2. If not installed, you will be redirected to the Microsoft Store. But use the official [website](https://www.python.org/downloads/windows/ "Python Releases for Windows") to download Python. **Scroll down and choose a 3.11 version!**
   3. Make a new directory PYVER for this Python version and use it in the installation process via "Customize installation". E.g. "C:\Users\NAME\PYVER\py3.11"
5. Install a new virtual environment:
   1. Open a PowerShell, cmd.exe or IDE.
   2. Change your directory to the project folder PYPROJ, e.g. with
   ``` cd PYPROJ ```
   3. Type and execute
   ``` C:\Users\NAME\PYVER\py3.11\python -m venv my_env ``` in a shell/command-line-interface.
6. Activate the environment:
   ```
   my_env\Scripts\activate
   ```
   If activated, the venv may be displayed before the prompt, like so: 
   
   ```(my_env) C:\Users\NAME\PYPROJ ```
7. Install the Python project:
   ``` 
   cd C:\Users\NAME\PYPROJ\smallmammalsclassifier\
   python -m pip install -e .
   ```
8. Now all functions are ready to use!


## Usage and recommended workflow

After installing, you can use all the functions in the project and the classifier on your data.

Additionally to the usage examples,  this section is structured and described as a recommended workflow.


### Split audio data into clips for prediction
The function **AudioSnippetWriter** takes 3 arguments.

```bash
AudioSnippetWriter [base_path] [--snippet_folder, -f] [--duration, -d]
```
| Argument             | Value                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| base_path            | (string) - Path to audio files                                                                                 |
| --snippet_folder, -f | (string, optional) - [default: creates folder in base_path] Path to where to store the created audio snippets. |
| --duration, -d       | (float, optional) - [default: 05:00] Duration of audio snippets in format mm:ss.                               |

**Example**

Execute the example in a shell with your variables:

```bash
# Create custom folder where to store the new audio files
$ mkdir /path/to/storage/recordings2024_snippets
# Execute the function with all parameters including the optional ones
$ AudioSnippetWriter /path/to/storage/recordings2024 -f /path/to/storage/recordings2024_snippets -d 10:00
```

### Classification with a pre-trained model

Before classifying all audio files, it is recommended to classify a test dataset. That is a small share of your audio data with annotation files for each audio files, i.e. selection tables from Raven.

#### Classify a test dataset ...

The function **MLClassification** takes 4 arguments.

```bash
MLClassification [model_path] [folder_path] [threshold] [--workers, -w]
```
| Argument      | Value                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| model_path    | (string) - Path to trained model, e.g. /path/to/PYPROJ/smallmammalsclassifier/resources/G2-1-long/                                               |
| folder_path   | (string) - Folder path to audio files to classify                                                                                                |
| threshold     | (float) - Decision threshold to round prediction scores. See *metrics.csv file for the selected model to decide which decision threshold to use. |
| --workers, -w | (int) - [default: 0] Number of workers for parallelization, i.e. cpus or cores. Default is 0 = current process.                                  |

**Example**

```bash
# To choose a number of workers execute this function. It will display your available CPUs.
$ GetCPU
# Classify test dataset with a chosen model
$ MLClassification /path/to/PYPROJ/smallmammalsclassifier/resources/G2-1-long/  /path/to/storage/testdataset_with_annotations/ 0.6 -w 2
```


#### ... and evaluate the results

The function **EvaluateTestDF** takes 3 arguments.

```bash
EvaluateTestDF [model_path] [--testdf_path, -t] [--no-display]
```
| Argument          | Value                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------- |
| model_path        | (string) - Path to trained model, e.g. /path/to/PYPROJ/smallmammalsclassifier/resources/G2-1-long/       |
| --testdf_path, -t | (string, optional) - Path to test_df.pkl and the prediction/classification of test_df from trained model |
| --no-display      | (flag, optional) - If no additional results from other thresholds should be displayed.                   |

**Example**

```bash
# Evaluate performance of chosen model on test dataset
# ! Important: classify the test dataset before evaluation!
$ EvaluateTestDF /path/to/PYPROJ/smallmammalsclassifier/resources/G2-1-long/ /path/to/PYPROJ/smallmammalsclassifier/temp/G2-1-long/ --no-display
```
If you are not sure whether the threshold suits your data, skip the flag 'no-display' to see the results for all thresholds, e.g. `EvaluateTestDF /path/to/PYPROJ/smallmammalsclassifier/resources/G2-1-long/ /path/to/PYPROJ/smallmammalsclassifier/temp/G2-1-long/`.


#### Classify all audio data

If the results from the test dataset classification are acceptable, execute the classification function on all your data.

**Example**

```bash
$ MLClassification /path/to/PYPROJ/smallmammalsclassifier/resources/G2-1-long/  /path/to/storage/recordings2024/ 0.6 -w 4
```

### Train a classifier

If the results from the test dataset are not acceptable, you can also train your own classifier with this project (or simply use the OpenSoundscape package and documentation for a new python project).


#### Prepare training files
The function **JSONWriter** writes json files with paths to the audio and annotation files used for training.
It takes 3 arguments.

```bash
JSONWriter [folder_in] [folder_out] [--suffix, -s]
```
| Argument         | Value                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| folder_in        | (string) - Folder path to audio data and Raven tables, see section [Dataset](#dataset) for a data structure example                  |
| --folder_out, -f | (string, optional) - [default: ./src/training/lib] Folder path where to write the json files.                                        |
| --suffix, -s     | (string, optional) - Common suffix of all Raven files created for the training audio files, e.g. *.Table.1.selections.txt (default). |

**Example**

```bash
# Prepare training data with your variables
$ JSONWriter /path/to/trainingdata/ -f /path/to/PYPROJ/smallmammalsclassifier/src/training/lib -s .Table.3.selections.txt
```

#### Training parameters
The function **SetParameters** writes a JSON file with the defined model parameters.
It takes 1 optional argument.

```bash
SetParameters [--folder_name, -f]
```
| Argument          | Value                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| --folder_name, -f | (string, optional) - [default: ./temp] Path to folder where to store the model parameter JSON files |
|                   |

The table shows all parameters with their default value.

| Parameter            | Default value     | Description                                                                                                                                                          |
| -------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| entity               | -                 | (string) 'Weights&Biases' parameter: username or team name where you're sending runs                                                                                 |
| project_name         | -                 | (string) 'Weights&Biases' parameter: Project name to be used while logging the experiment with wandb                                                                 |
| model_name           | test              | (string) Name of the model, e.g. 'G2-1-long' or 'myCustomModel'                                                                                                      |
| run_name             | -                 | (string) 'Weights&Biases' parameter: Experiment/ run name to be used while logging the training process                                                              |
| group_name           | -                 | (string) 'Weights&Biases' parameter: Name of the group associated with the run                                                                                       |
| epochs_to_train      | 100               | (int) Number of steps/epochs for the training iteration                                                                                                              |
| save_interval        | 1                 | (int) interval in epochs to save model object with weights, 1= save every epoch. The best model is always saved to best.model in addition to other saved epochs!     |
| num_workers          | 0                 | (int) Defines the number of kernels to use for training; 0 = root process                                                                                            |
| train_size           | 0.7               | (float) Proportion to split data in training and validation datasets for internal validation                                                                         |
| clip_duration        | 1.2               | (float) Duration in seconds of clips (training instances)                                                                                                            |
| n_samples            | 3                 | (int) Number of clip samples to inspect before training starts                                                                                                       |
| channels             | 3                 | (int) Number of channels of sample shape; defines input and output of samples                                                                                        |
| min_fr               | 0                 | (int) Low frequency in Hz for bandpassing                                                                                                                            |
| display_columns      | 3                 | (int) Number of columns to display the samples during sample inspection                                                                                              |
| batch_size           | 32                | (int) Number of instances loaded simultaneously in each batch during a training epoch, 'Rule of thumb' = 32; a higher batch sizes like 64 or 128 require more memory |
| bypass_augmentations | False             | (bool) Whether to bypass augmentation of samples before training; False = added augmentation, True = no augmentation                                                 |
| scale                | groups            | (string) Class scale to use; groups or genera (see ./src/shared/genera.py and ./src/shared/groups_minimal.py for included species names)                             |
| folder_jsons         | src/training/lib/ | (string) Path to JSON files with training data (see [Prepare training files](#prepare-training-files))                                                               |
| architecture         | resnet18          | (string) Architecture of a CNN                                                                                                                                       |
| imgsize              | 448               | (int) Size of input images of samples in (height,width).                                                                                                             |
| sampling_rate        | 192000            | (int) Target sample rate in Hz. Should not exceed the sampling rate of the audio files.                                                                              |


**Example**

```bash
# Set or accept default training parameters
$ SetParameters
# Follow the instructions shown in the shell. Accept the default parameters or type a new value for each parameter in the prompt when asked to do so.
```


## Dataset

The dataset used for training the provided models includes audio data of various small terrestrial mammals. The dataset itself is not included in this repository for storage and licensing reasons. However, you can find available data under [BTO (British Trust of Ornithology)](https://www.bto.org/our-science/publications/peer-reviewed-papers/acoustic-identification-small-terrestrial-mammals "The acoustic identification of small terrestrial mammals in Britain") and [ChiroVox](https://www.chirovox.org/ "The bat call library").
Additionally, if you need the data I annotated with a coarse class scale (small terrestrial mammals, noise, and bats), please contact me at git@mariariffel.de or the Chair of Geobotany at the University of Freiburg, Germany.

Please make sure to follow the annotation scheme as in ./src/shared/groups_minimal.py or ./src/shared/genera.py, or simply add your own annotations to the files to match the class scales.

To train the model on your own dataset, please confirm that your audio and Raven data can be found in the same folder, like so:

```
dataset/
├── species_1/
│   ├── audio1.WAV
│   ├── audio1.Table.1.selections.txt
│   ├── audio2.WAV
│   ├── audio2.Table.1.selections.txt
├── species_2/
│   ├── audio1.WAV
│   ├── audio1.Table.1.selections.txt
│   ├── audio2.WAV
│   ├── audio2.Table.1.selections.txt
...
```


## Models

The models provided are built using the OpenSoundscape library (internally uses PyTorch). They are trained on an audio dataset of small terrestrial mammal calls.
The model G2-1-long achieves approximately 25% target recall with an F1-Score of 0.23 in identifying the species groups of small terrestrial mammals.
The model G2-6-long achieves approxiamtely 79% target recall.
You can find details for each model in "resources".

If you want to train your own model, please adjust the parameter with the SetParameters script before executing train.py.

## Contributing

Contributions are welcome :)
If you have ideas for improving the code, the classifier, adding new species, or enhancing recall, precision, or F1, please submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
