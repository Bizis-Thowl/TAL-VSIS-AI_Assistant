# Testsystem: "missy", Live-System: "incluedo-mis"
system_spec = "incluedo-mis"
# Base URL 1 is for the retrieval of information from Missy
base_url_missy = f"https://{system_spec}.evabrain.de/webservice-1/"
# Base URL 2 is for the interface with the AI assistant
base_url_ai = f"https://{system_spec}.evabrain.de/webservice-2/"

log_store = "store"

update_cache = False

relevant_date_test = "2025-01-10"

training_features = ["timeToSchool", "cl_experience", "short_term_cl_experience", "school_experience", "priority", "ma_availability", "mobility", "geschlecht_relevant", "qualifications_met"]

training_features_de = ["Zeit bis zur Schule", "Kundenbeziehung", "Kundenbeziehung kurzfristig", "Schulbeziehung", "Priorität", "MA-Verfügbarkeit", "Mobilität", "Geschlecht relevant", "Qualifikationen erfüllt"]