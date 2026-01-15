from typing import List, Tuple, Dict

from data_processing.features_retrieval.client_features import aggregate_client_features
from data_processing.features_retrieval.ma_features import aggregate_ma_features
from data_processing.features_retrieval.filter_mabw_records import filter_mabw_records
from data_processing.features_retrieval.filter_kabw_records import filter_kabw_records
from data_processing.features_retrieval.retrieve_ids import (
    get_ma_assignments,
    get_open_client_ids,
    get_free_ma_ids,
)
from data_processing.features_retrieval.retrieve_objects import get_objects_by_id


class DataProcessor:

    def __init__(
        self,
        mas,
        clients,
        prio_assignments,
        distances,
        experience_log,
        global_schools_mapping,
    ) -> None:
        self.mas = mas
        self.clients = clients

        self.prio_assignments = prio_assignments
        self.distances = distances
        self.experience_log = experience_log
        self.global_schools_mapping = global_schools_mapping

    def get_mabw_records(self, vertretungen: List) -> Dict:

        filtered_mabw_records = filter_mabw_records(vertretungen)

        return filtered_mabw_records

    def get_kabw_records(self, vertretungen: List, assigned_mas: List) -> Dict:

        filtered_kabw_records = filter_kabw_records(vertretungen, assigned_mas)

        return filtered_kabw_records

    def get_ma_assignments(self, rescheduled_ma_records: List) -> Dict:

        return get_ma_assignments(rescheduled_ma_records)

    def create_day_dataset(self, clients, mas, date: str):

        open_client_objects = get_objects_by_id(self.clients, clients)
        free_ma_objects = get_objects_by_id(self.mas, mas)

        clients_df, clients_dict = aggregate_client_features(
            open_client_objects,
            date,
            self.prio_assignments,
            self.global_schools_mapping,
        )
        mas_df, mas_dict = aggregate_ma_features(
            free_ma_objects,
            self.distances,
            clients_dict,
            self.experience_log,
            date.strftime("%Y-%m-%d"),
            self.global_schools_mapping,
        )

        return clients_df, mas_df

    def get_open_clients_and_mas(
        self, vertretungen: List, assigned_mas: List, open_client_records: List
    ) -> Tuple[List, List]:

        filtered_kabw_records = filter_kabw_records(vertretungen, assigned_mas)
        absent_client_records = filtered_kabw_records["absent_clients"]
        free_ma_records = filtered_kabw_records["free_mas"]

        open_client_ids = get_open_client_ids(open_client_records)
        free_ma_ids = get_free_ma_ids(free_ma_records, absent_client_records, self.mas)

        return open_client_ids, free_ma_ids

    def get_client_record_assignments(self, records: List) -> Dict[str, Dict]:

        client_record_assignments = {}

        for record in records:
            client_record_assignments[record["klientzubegleiten"]["id"]] = {
                "id": record["id"],
                "org": record.get("org"),
            }

        return client_record_assignments
