import numpy as np
import cmdstanpy as cmd
import pandas as pd
import pickle
from joblib import Parallel, delayed
from typing import Dict, Optional, List, Tuple, Union, Iterable


class BG_NBD:
    def __init__(self, stan_file_path: str, seed: Optional[int] = None):
        self._model = cmd.CmdStanModel(stan_file=stan_file_path)
        self._seed = seed if seed is not None else 1234
        self._data_dict = None
        self._customer_id_to_index = None
        self._samples = None
        self._draws = None
        self.predictive_distribution_train = None
        self.random_state = np.random.default_rng(seed=self._seed)

    @property
    def config(self):
        return {
            "seed": self._seed,
            "model": self._model,
        }

    @property
    def fit_config(self):
        return {
            "data_dict": self._data_dict,
            "customer_id_to_index": self._customer_id_to_index,
            "samples": self._samples,
            "draws": self._draws,
        }

    def fit(
        self, df: pd.DataFrame, prior_only: bool = False, **stan_sample_kwargs
    ) -> None:
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
            "T_age": df["T"].astype(int).to_numpy(),
        }
        self._customer_id_to_index = _generate_customer_index_lookup(df)
        self._samples = self.config["model"].sample(
            self._data_dict, seed=self.config["seed"], **stan_sample_kwargs
        )
        self._draws = self._samples.draws(concat_chains=True)

    def simulate_for_customer_id(
        self, customer_id: int
    ) -> Dict[Tuple[int, List[int]], List[List[float]]]:
        """Simulate from the prior/posterior predictive distribution for a given customer id, from time 0.

        Args:
            customer_id (int): Customer id/row identifier.

        Raises:
            AttributeError: if .fit() isn't run first, will raise an error.

        Returns:
            Dict[int, List[List[float]]]: dictionary of simulated arrival times of transactions for the given customer_id.
        """
        if self.fit_config["samples"] is None:
            raise AttributeError("No samples found. Run .fit() on data first.")
        customer_specific_params = _get_parameters_for_customer(
            customer_id,
            self.fit_config["customer_id_to_index"],
            self.fit_config["draws"],
            self.fit_config["samples"].column_names,
        )
        sims = []
        for p, lamb in zip(
            customer_specific_params["p"],
            customer_specific_params["lambda"],
            strict=True,
        ):
            arrival_time_sims = []
            time = self.random_state.exponential(1 / lamb)
            alive = True
            while alive:
                arrival_time_sims.append(time)
                next_time_to_arrival = self.random_state.exponential(1 / lamb)
                time += next_time_to_arrival
                alive = self.random_state.uniform(0, 1) >= p
            sims.append(arrival_time_sims)
        return sims

    def predict(
        self,
        n_jobs: int = 1,
        newdata: Optional[pd.DataFrame] = None,
    ) -> Dict[int, List[List[float]]]:
        """Draw from the prior/posterior predictive distribution from time 0.

        Args:
            n_jobs (int, optional): Number of jobs when simulating in parallel. Defaults to 1.
            newdata (Optional[pd.DataFrame], optional): pandas dataframe that contains customer ids. If customer ids cannot be found in the training dataset,
            simulations are drawn at the population level. Defaults to None.

        Returns:
            Dict[int, List[List[float]]]: dictionary with customer ids as keys and lists of simulated transactions as values
        """
        customer_keys_train = self.fit_config["customer_id_to_index"].keys()
        if newdata is None:
            if self.predictive_distribution_train is None:
                all_sims = Parallel(n_jobs=n_jobs)(
                    delayed(lambda x: self.simulate_for_customer_id(x))(idx)
                    for idx in customer_keys_train
                )
                self.predictive_distribution_train = {
                    idx: sim
                    for idx, sim in zip(customer_keys_train, all_sims, strict=True)
                }
                return self.predictive_distribution_train
            else:
                return self.predictive_distribution_train
        # TODO: if newdata, check if ids are in customer_id_to_index. If not, then simulate those observations from the population parameters.

    @staticmethod
    def filter_sims_between(
        min_time_weeks: Union[int, Iterable],
        sims: Dict[int, List[List[float]]],
        duration_weeks: int = 100,
    ) -> Dict[int, List[List[float]]]:
        """Filters a dictionary of simulations between specific points in time.

        Args:
            min_time_weeks (Union[int, List[int]]): filter simulations for all transactions after this point. If given as a list,
                changes the cutoff for each customer id in sims (in order).
            sims (Dict[int, List[List[float]]]): dictionary of simulations for each customer, the result of running .predict()
            duration_weeks (int): how long to run the simulation for. I.e. if min_time_weeks is specified, the
                max time that transactions are simulated to is min_time_weeks + duration_weeks. Default is 100.

        Returns:
            Dict[int, List[List[float]]]: copy of sims with transactions that occur only after min_time_weeks.
        """
        min_time_weeks = (
            [min_time_weeks] * len(sims)
            if isinstance(min_time_weeks, int)
            else min_time_weeks
        )
        time_cutoffs = [
            (x, x + y)
            for x, y in zip(min_time_weeks, [duration_weeks] * len(sims), strict=True)
        ]

        new_sims = {}
        for (customer, sims), (min_t, max_t) in zip(
            sims.items(), time_cutoffs, strict=True
        ):
            filtered_sims = []
            for s in sims:
                filtered_sims.append(
                    [time for time in s if time >= min_t and time <= max_t]
                )
            new_sims[customer] = filtered_sims
        return new_sims

    def diagnostics(self):
        return self.fit_config["samples"].diagnose()

    def summary_table(
        self, sort_by: Optional[List[str]] = "R_hat", ascending: bool = False
    ):
        return (
            self.fit_config["samples"]
            .summary()
            .sort_values(by=sort_by, ascending=ascending)
        )

    def save(self, directory: str) -> None:
        with open(directory, "wb") as f:
            pickle.dump(self, f)
        return None


def _get_parameters_for_customer(
    customer_id: int,
    customer_index_lookup: Dict[int, int],
    draws: np.ndarray,
    sample_column_names: Tuple[str],
) -> Dict[str, np.ndarray]:
    id_lookup = f"[{customer_index_lookup[customer_id]}]"
    p_array_position = [
        idx for idx, name in enumerate(sample_column_names) if f"p{id_lookup}" in name
    ]
    lambda_array_position = [
        idx
        for idx, name in enumerate(sample_column_names)
        if f"lambda{id_lookup}" in name
    ]
    return {
        "p": draws[:, p_array_position].flatten(),
        "lambda": draws[:, lambda_array_position].flatten(),
    }


def _generate_customer_index_lookup(df: pd.DataFrame):
    return {
        customer_id: customer_index + 1
        for customer_id, customer_index in zip(
            df["customer_id"], range(0, df.shape[0]), strict=True
        )
    }
