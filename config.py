# Base URL 1 is for the retrieval of information from Missy
base_url_missy = "https://{domain}.evabrain.de/webservice-1/"
# Base URL 2 is for the interface with the AI assistant, e.g. system_spec = "missy"
base_url_ai = "https://{system_spec}.evabrain.de/webservice-2/"

log_store = "store"

include_abnormality = False

# Set to False for a live test today
relevant_date_test = "2025-11-25"

training_features = ["timeToSchool", "cl_experience", "short_term_cl_experience", "school_experience", "priority", "ma_availability", "mobility", "geschlecht_relevant", "qualifications_met", "availability_gap"]

training_features_de = ["Zeit bis zur Schule", "Kundenbeziehung", "Kundenbeziehung kurzfristig", "Schulbeziehung", "Priorität", "MA-Verfügbarkeit", "Mobilität", "Geschlecht relevant", "Qualifikationen erfüllt", "Verfügbarkeitsabstand"]