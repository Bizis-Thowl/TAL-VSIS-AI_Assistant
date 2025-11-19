
## Setup (For Linux - also deployable on Windows with small adaptations)

### Install UV for Virtual Environment (Recommended)

Full installation instructions for UV: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

``curl -LsSf https://astral.sh/uv/install.sh | sh``

### Install Supported Python Version (3.11)

``uv python install 3.11``

### Clone Git Repository

``git clone https://github.com/Bizis-Thowl/TAL-VSIS-AI_Assistant.git``

### Setup Virtual Environment

``cd TAL-VSIS-AI_Assistant``\
``uv venv .venv``\
``source .venv/bin/activate``\
``uv pip install -r requirements``

### Setup config and .env

#### config

The configuration can be adapted for the live- or test-system via
the variables. Additionally, it can be specified a specific date for the run, or alternatively provided `relevant_test_date = False`

In the first run, the relatively static data is fetched. Whenever changes occur to the basic data, the cache should be updated (delete + restart).

#### .env

There is a file called `.env_example` create a copy of it and name it `.env`. Then fill in the necessary entries (user and password).

### Store and Update Necessary Data

To save all vertretungsfälle from a starting date (The date in the script must be adapted as needed. Currently defaults to 20.02.25) to the current date:
```
python test2.py
```
Extract necessary information from the vertretungsfälle and prepare the data to be used in the main script:
```
python update_ma_client_history.py
```

### Train Machine Learning Model (iForest)

```
python train_iforest.py
```

## Running the Code (Continuously)

```
python main.py
```

## Bonus

### Analysis

To run a basic data analysis you can run the jupyter notebook `analysis.ipynb`. The file `analysis.py` contains further methods that can be used and adapted in the notebook, as well as extended for deeper and broader data analysis.