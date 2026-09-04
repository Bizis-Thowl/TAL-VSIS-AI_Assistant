# TALÖVSIS AI Assistant

**TALÖVSIS** steht für *Trägerübergreifendes, autonom lernendes, ÖPNV-berücksichtigendes Vertretungsmanagement-System für Inklusion – Schulbegleitung*. Das Forschungsprojekt entwickelt einen KI-gestützten Demonstrator, der Träger der Schulbegleitung bei der kurzfristigen Organisation von Vertretungen unterstützt. Auf Basis verfügbarer Mitarbeitender und offener Begleitbedarfe werden geeignete Zuordnungen vorgeschlagen. Dabei berücksichtigt das System unter anderem Qualifikation und Eignung, zeitliche Verfügbarkeit, Mobilität und Reisezeiten, Entfernung sowie bisherige Einsatzerfahrungen.

Dieses Repository enthält den KI- und Optimierungskern des TALÖVSIS-Demonstrators. Der Python-Code ruft relevante Planungsdaten aus den angebundenen Fachsystemen ab, bereitet sie auf und erzeugt unter Berücksichtigung fachlicher Randbedingungen mehrere mögliche Vertretungszuordnungen. Ergänzende lernende und erklärende Komponenten dienen dazu, ungewöhnliche Zuordnungen zu erkennen und die erzeugten Empfehlungen nachvollziehbar bereitzustellen. Das Repository bildet damit insbesondere das Backend des AI Assistant ab und nicht die vollständige Fachanwendung beziehungsweise deren Benutzeroberfläche.

## Setup (For Linux - also deployable on Windows with small adaptations)



### Install UV for Virtual Environment (Recommended)

Full installation instructions for UV: [https://docs.astral.sh/uv/getting-started/installation/#standalone-installer](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

`curl -LsSf https://astral.sh/uv/install.sh | sh`

### Install Supported Python Version (3.11)

`uv python install 3.11`

### Clone Git Repository

`git clone https://github.com/Bizis-Thowl/TAL-VSIS-AI_Assistant.git`

### Setup Virtual Environment

`cd TAL-VSIS-AI_Assistant`  
`uv venv .venv`  
`source .venv/bin/activate`  
`uv pip install -r requirements`

### Setup config and .env



#### config

The configuration can be adapted for the live- or test-system via
the variables. Additionally, it can be specified a specific date for the run, or alternatively provided `relevant_test_date = False`

In the first run, the relatively static data is fetched. Whenever changes occur to the basic data, the cache should be updated (delete + restart).

#### .env

There is a file called `.env_example` create a copy of it and name it `.env`. Then fill in the necessary entries (REQUEST_INFO).

### Store and Update Necessary Data

To save all vertretungsfälle from a starting date to an end date. Also, extract necessary information from the vertretungsfälle and prepare the data to be used in the main script:

```
python test2.py
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