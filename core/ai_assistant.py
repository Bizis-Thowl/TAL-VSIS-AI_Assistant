from typing import Dict, List, Tuple, Optional
import pandas as pd

from core.data.data_manager import DataManager
from core.optimization.optimization_engine import OptimizationEngine
from core.learning.learning_engine import LearningEngine
from core.recommendation.recommendation_manager import RecommendationManager


class AIAssistant:
    """Main class coordinating the staff planning system."""
    
    def __init__(
        self,
        user: str,
        pw: str,
        mas: List[Dict],
        clients: List[Dict],
        prio_assignments: Dict,
        distances: Dict,
        experience_log: Dict
    ):
        self.data_manager = DataManager(
            user, pw, mas, clients, prio_assignments, distances, experience_log
        )
        self.optimization_engine = OptimizationEngine()
        self.learning_engine = LearningEngine()
        self.recommendation_manager = RecommendationManager(user, pw)

    def get_vertretungen(self, date_str: Optional[str] = None) -> List[Dict]:
        """Get vertretungen data."""
        return self.data_manager.get_vertretungen(date_str)

    def get_dataset(self, vertretungen: List[Dict], date_str: Optional[str] = None) -> bool:
        """Update the dataset with new vertretungen data."""
        did_update, client_record_assignments, clients_df, mas_df = self.data_manager.update_dataset(
            vertretungen, date_str
        )
        
        if did_update:
            self.optimization_engine.update_model(clients_df, mas_df)
            self.learning_engine.update_data(clients_df, mas_df)
            
        return did_update

    def update_solver_model(self) -> None:
        """Update the optimization model."""
        if not self.optimization_engine.is_initialized():
            print("Please update the dataset first")
            return
            
        self.optimization_engine.update_model(
            self.data_manager.get_clients_df(),
            self.data_manager.get_mas_df()
        )

    def solve_model(self, objective_value: Optional[float] = None) -> Optional[float]:
        """Solve the optimization model."""
        if not self.optimization_engine.is_initialized():
            print("Please update the solver model first")
            return None
            
        new_objective, assigned_pairs, recommendation_id = self.optimization_engine.solve(
            objective_value
        )
        return new_objective

    def process_results(self) -> Tuple[List[Dict], str]:
        """Process the optimization results."""
        if not self.optimization_engine.is_initialized():
            print("Please solve the model first")
            return [], ""
            
        _, assigned_pairs, recommendation_id = self.optimization_engine.solve()
        return assigned_pairs, recommendation_id

    def prepare_learner_data(self, assignments: Dict) -> List[List]:
        """Prepare data for the learning engine."""
        if not self.learning_engine.is_initialized():
            print("Please update the dataset first")
            return []
            
        return self.learning_engine.prepare_data(assignments)

    def retrieve_learner_scores(self, datapoint: List[List]) -> Tuple[int, float]:
        """Get scores from the learning engine."""
        if not self.learning_engine.is_initialized():
            print("Please update the dataset first")
            return 0, 0.0
            
        return self.learning_engine.predict_and_score(datapoint)

    def send_update(
        self,
        assigned_pairs: List[Dict],
        recommendation_ids: List[str],
        learner_infos: List[Tuple[int, float]]
    ) -> Dict:
        """Send recommendation updates."""
        return self.recommendation_manager.send_update(
            assigned_pairs,
            recommendation_ids,
            learner_infos,
            self.data_manager.get_client_record_assignments()
        ) 