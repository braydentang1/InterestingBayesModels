import pickle
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cmdstanpy as cmd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit


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
        self, customer_id: int, seed: int
    ) -> Dict[Tuple[int, List[int]], List[List[float]]]:
        """Simulate from the prior/posterior predictive distribution for a given customer id, from time 0.

        Args:
            customer_id (int): Customer id/row identifier.
            seed (int): integer seed to simulate parameters when a customer id is not found in the training data.

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
            seed,
        )
        sims = []
        number_of_transactions = self.random_state.geometric(
            customer_specific_params["p"]
        )
        lambda_repeated = np.repeat(
            customer_specific_params["lambda"], number_of_transactions
        )
        arrival_times = self.random_state.exponential(1 / lambda_repeated)
        start = 0
        for n in number_of_transactions:
            sims.append(np.cumsum(arrival_times[start : (start + n)]))
            start += n
        return sims

    def predict(
        self,
        newdata: pd.DataFrame,
        n_jobs: int = 1,
    ) -> Dict[int, List[List[float]]]:
        """Draw from the prior/posterior predictive distribution from time 0.

        Args:
            newdata (pd.DataFrame): pandas dataframe that contains a column called customer_id. If a specific customer id
             cannot be found in the training dataset, simulations are drawn at the population level.
            n_jobs (int): Number of jobs when simulating in parallel. Defaults to 1.

        Returns:
            Dict[int, List[List[float]]]: dictionary with customer ids as keys and lists of simulated transactions as values
        """
        customer_keys = set(newdata["customer_id"])
        seeds = self.random_state.integers(low=1, high=1e9, size=len(customer_keys))
        all_sims = Parallel(n_jobs=n_jobs)(
            delayed(lambda x, y: self.simulate_for_customer_id(x, y))(idx, seed)
            for idx, seed in zip(list(customer_keys), seeds)
        )
        return _sims_to_dict(list(customer_keys), all_sims)

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
    seed: int,
) -> Dict[str, np.ndarray]:
    if customer_index_lookup.get(customer_id) is not None:
        id_lookup = f"[{customer_index_lookup[customer_id]}]"
        p_lambda_array_positions = [
            _find_parameter_position(f"{variable}{id_lookup}", sample_column_names)
            for variable in ["p", "lambda"]
        ]
        return {
            "p": draws[:, p_lambda_array_positions[0]].flatten(),
            "lambda": draws[:, p_lambda_array_positions[1]].flatten(),
        }
    else:
        rng = np.random.default_rng(seed=seed)
        parameters = [
            _find_parameter_position(f"{variable}", sample_column_names)
            for variable in ["gamma_alpha", "gamma_beta", "p_logit_mu", "p_logit_sigma"]
        ]
        # Draw new lambda and p
        new_lambda = rng.gamma(
            draws[:, parameters[0]],
            draws[:, parameters[1]],
        )
        new_p = expit(
            rng.normal(
                draws[:, parameters[2]],
                draws[:, parameters[3]],
            )
        )

        return {"p": new_p.flatten(), "lambda": new_lambda.flatten()}


def _generate_customer_index_lookup(df: pd.DataFrame) -> Dict[int, int]:
    return {
        customer_id: customer_index + 1
        for customer_id, customer_index in zip(
            df["customer_id"], range(0, df.shape[0]), strict=True
        )
    }


def _find_parameter_position(
    variable_name: str, sample_column_names: Tuple[str]
) -> List[int]:
    return [
        idx for idx, name in enumerate(sample_column_names) if variable_name in name
    ]


def _sims_to_dict(
    customer_ids: List[int], all_sims: List
) -> Dict[int, List[List[float]]]:
    return {idx: sim for idx, sim in zip(customer_ids, all_sims, strict=True)}
