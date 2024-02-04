import numpy as np
import cmdstanpy as cmd
import pandas as pd
import pickle 
from typing import Dict, Optional, List, Tuple


class BG_NBD:
    def __init__(self, stan_file_path: str, seed: Optional[int] = None):
        self._model = cmd.CmdStanModel(stan_file=stan_file_path)
        self._seed = seed if seed is not None else 1234
        self._data_dict = None
        self._customer_id_to_index = None
        self._samples = None
        self._draws = None
        self.random_state = np.random.default_rng(seed=self._seed)
    
    @property
    def config(self):
        return {
            "data_dict": self._data_dict,
            "seed": self._seed,
            "customer_id_to_index": self._customer_id_to_index,
            "samples": self._samples,
            "draws": self._draws,
            "model": self._model,
            }

    def fit(self, df: pd.DataFrame, prior_only: bool = False, **stan_sample_kwargs) -> None:
        """Runs the MCMC sampler for the BG-NBD model.

        Args:
            df (pd.DataFrame): Pandas dataframe containing named columns recency, frequency, and T.
            prior_only (bool): Sample from the prior only (True)?
            **stan_sample_kwargs: any additional keyword arguments to sample() of cmdstanpy
        """
        self._data_dict = {
            "prior_only": 1 if prior_only else 0,
            "N_customers": df.shape[0],
            "recency": df["recency"].astype(int).to_numpy(),
            "frequency": df["frequency"].astype(int).to_numpy(),
            "T_age": df["T"].astype(int).to_numpy()
        }
        self._customer_id_to_index = _generate_customer_index_lookup(df)
        self._samples = self.config["model"].sample(
            self._data_dict,
            seed=self.config["seed"],
            **stan_sample_kwargs
        )
        self._draws = self._samples.draws(concat_chains=True)
    
    def simulate_for_customer_id(
        self,
        customer_id: int,
        max_time_weeks: Optional[int] = None
        ) -> Dict[Tuple[int, List[int]], List[List[float]]]:
        """Simulate from the prior/posterior predictive distribution for a given customer id, from time 0.

        Args:
            customer_id (int): Customer id/row identifier.
            max_time_weeks (Optional[int]): Optional max number of weeks in the future to simulate.

        Raises:
            AttributeError: if .fit() isn't run first, will raise an error.

        Returns:
            Dict[int, List[List[float]]]: dictionary of simulated arrival times of transactions for the given customer_id.
        """
        if self.config["samples"] is None:
            raise AttributeError("No samples found. Run .fit() on data first.")
        self.simulations = {}
        if self.simulations.get(customer_id) is not None:
            return self.simulations[customer_id]
        else:
            customer_specific_params = _get_parameters_for_customer(
                customer_id,
                self.config["customer_id_to_index"],
                self.config["draws"],
                self.config["samples"].column_names,
            )
            sims = []
            for p, lamb in zip(customer_specific_params["p"], customer_specific_params["lambda"], strict=True):
                arrival_time_sims = []
                time = self.random_state.exponential(1/lamb)
                alive = True
                while alive:
                    arrival_time_sims.append(time)
                    next_arrival_time = self.random_state.exponential(1/lamb)
                    time += next_arrival_time
                    alive = self.random_state.uniform(0, 1) >= p
                if max_time_weeks is not None:
                    sims.append([time for time in arrival_time_sims if time <= max_time_weeks])
                else:
                    sims.append(arrival_time_sims)
                self.simulations[customer_id] = sims
            return self.simulations[customer_id]

    def diagnostics(self):
        return self.config["samples"].diagnose()
    
    def summary_table(self, sort_by: Optional[List[str]] = "R_hat", ascending: bool = False):
        return self.config["samples"].summary().sort_values(by=sort_by, ascending=ascending)

    def save_samples(self, directory: str) -> None:
        with open(directory, "wb") as f:
            pickle.dump(self.config["samples"], f)
        return None


def _get_parameters_for_customer(
    customer_id: int,
    customer_index_lookup: Dict[int, int],
    draws: np.ndarray, 
    sample_column_names: Tuple[str],
    ) -> Dict[str, np.ndarray]:
    id_lookup = f"[{customer_index_lookup[customer_id]}]"
    p_array_position = [idx for idx, name in enumerate(sample_column_names) if f"p{id_lookup}" in name]
    lambda_array_position = [idx for idx, name in enumerate(sample_column_names) if f"lambda{id_lookup}" in name]
    return {
        "p": draws[:, p_array_position].flatten(),
        "lambda": draws[:, lambda_array_position].flatten()
    }

def _generate_customer_index_lookup(df: pd.DataFrame):
    return {
        customer_id: customer_index + 1
        for customer_id, customer_index in 
        zip(df["customer_id"], range(0, df.shape[0]), strict=True)
    }