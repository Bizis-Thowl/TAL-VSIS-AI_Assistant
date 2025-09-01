
## Setup (For Linux - also deployable on Windows with small adaptations)

### Install UV for Virtual Environment (Recommended)

Full installation instructions: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

``curl -LsSf https://astral.sh/uv/install.sh | sh``

### Install Supported Python Version (3.11)

``uv python install 3.11``

### Clone Git Repository

``git clone https://github.com/Bizis-Thowl/TAL-VSIS-AI_Assistant.git``

### Setup Virtual Environment

``cd TAL-VSIS-AI_Assistant``\
``uv venv .venv``\
``source .venv/bin/activate``
```

```

### Setup config and .env

#### config

The configuration can be adapted for the live- or test-system via
the variables. Additionally, it can be specified a specific date for the run, or alternatively provided `relevant_test_date = False`

For the first run it is necessary to change the config as follows:

```
    update_cache = True
```

After the first run, the relatively static data is fetched and the value can be set to <i>False</i> again. Whenever changes occur to the basic data, the cache should be updated.

#### .env

There is a file called `.env_example` create a copy of it and name it `.env`. Then fill in the necessary entries (user and password).

### Store and Update Necessary Data

```
    # to save vertretungsfälle from a starting date (The date in the script must be adapted as needed. Currently defaults to 20.05.25)
    python test2.py
    # extract necessary information from the vertretungsfälle and prepare the data to be used in the main script
    python update_ma_client_history.py
```

## Running the Code

Entrypoint: ./main.py