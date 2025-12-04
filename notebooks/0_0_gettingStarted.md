# Getting Started
These sections establish basic libaries that we'll be using for data handling and plotting, as well as the initial data acquisition and reformatting for analysis.

## Directory Setup

This project requires a `data` directory to store downloaded files and an `output` directory to save results like plots and tables.

You have two options for setting up these directories:

**Option 1 (Recommended): Create a `config_env.py` file**

For a persistent setup, create a file named `config_env.py` inside the `notebooks/` directory. This file will tell the notebooks where to find your data and save your outputs. This file is ignored by version control.

Your `config_env.py` file should look like this. Copy what's below and replace the example paths with your own desired locations.

```python
# In notebooks/config_env.py
# You can use absolute paths, or relative paths from the project root.
DATA_DIR = "/path/to/your/data"
OUTPUT_DIR = "/path/to/your/output"
```

**Option 2: Do Nothing**

If you don't create a `config_env.py` file, the notebooks will automatically create and use `data/` and `output/` subdirectories within the project's root folder. This is the simplest way to get started.


```{tableofcontents}
```
