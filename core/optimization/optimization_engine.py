from typing import Dict, List, Tuple, Optional
import pandas as pd

from optimize.optimize import Optimizer


class OptimizationEngine:
    """Handles constraint optimization for staff planning."""
    
    def __init__(self):
        self.optimizer = None
        self.clients_df = pd.DataFrame()
        self.mas_df = pd.DataFrame()

    def update_model(self, clients_df: pd.DataFrame, mas_df: pd.DataFrame) -> None:
        """Update the optimization model with new data."""
        self.clients_df = clients_df
        self.mas_df = mas_df
        
        if self.optimizer is None:
            self.optimizer = Optimizer(self.mas_df, self.clients_df)
            self.optimizer.create_model()
        else:
            self.optimizer.create_model()

    def solve(self, objective_value: Optional[float] = None) -> Tuple[Optional[float], List[Dict], str]:
        """Solve the optimization model and return results."""
        if self.optimizer is None:
            return None, [], ""
            
        new_objective = self.optimizer.solve_model(objective_value)
        # Add 10% to encourage diversity
        new_objective = int(new_objective * 0.9)
        
        assigned_pairs, recommendation_id = self.optimizer.process_results()
        return new_objective, assigned_pairs, recommendation_id

    def is_initialized(self) -> bool:
        """Check if the optimization engine is properly initialized."""
        return self.optimizer is not None and not self.clients_df.empty and not self.mas_df.empty 