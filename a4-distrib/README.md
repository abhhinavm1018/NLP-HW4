# CSE 378: Recurrent Neural Networks, Attention, and Reading Comprehension

## Table of Contents

- [Installation](#installation)
- [Downloading the data](#downloading-the-data)

## Installation

[Conda](https://conda.io/) will set up a virtual environment with the exact version of Python
used for development along with all the dependencies needed to run the code. 
You could also use python built-in [virtual environments](https://docs.python.org/3/library/venv.html).

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Change your directory to your clone of this repo.

    ```
    cd cs378_RNN
    ```

3.  Create a Conda environment with Python 3. (You can also specify other versions you like; the code has been tested on Python3.6 and 3.9.)

    ```
    conda create -n cs378_RNN python=3.9
    ```

4.  Now activate the Conda environment.
    You will need to activate the Conda environment in each terminal in which you 
    want to run code from this repo.

    ```
    source activate cs378_RNN
    ```

5.  Install the required dependencies.

    ```
    pip install -r requirements.txt
    ```
    

You should now be able to test your installation with `pytest -v`.  Congratulations!

## Downloading the data

By default, the code expects SQuAD data in a folder at `./squad/`, with files 
`train_small.json`, `val_small.json`, and `test_small.json`.

In addition, the code expects `glove.6B.50d` vectors in `./glove/`. You can download
these vectors from the [GloVe website @ StanfordNLP](https://nlp.stanford.edu/projects/glove/) ---
`glove.6B.zip` is the archive you want, and unzipping it will give you the vectors. Feel free to
experiment with using the other GloVe vectors as well!

