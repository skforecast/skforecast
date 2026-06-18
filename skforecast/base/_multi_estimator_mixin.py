################################################################################
#                             MultiEstimatorMixin                              #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import pandas as pd

from ..utils import check_select_fit_kwargs


class MultiEstimatorMixin:
    """
    Mixin that adds management of multiple estimators to a forecaster.

    This mixin is not meant to be instantiated on its own. It provides methods
    to inspect and remove estimators in forecasters that hold more than one
    estimator. The class using this mixin must initialize the following
    attributes in its `__init__` method:

    - `estimators` : list
        Original (unfitted) estimator instances provided by the user.
    - `estimators_` : list
        Estimator instances that are trained when `fit()` is called.
    - `estimator_ids` : list
        Unique identifier for each estimator.
    - `estimator_names_` : list
        Descriptive name for each estimator.
    - `estimator_types` : list
        Full qualified type string for each estimator.
    - `estimator_params_` : dict
        Parameters of each estimator keyed by estimator id. Required by
        `get_estimators_info`.
    - `n_estimators` : int
        Number of estimators in the forecaster.

    The following attributes are optional. When present, `get_estimators_info`
    adds the corresponding column to its output:

    - `estimators_support_exog` : tuple
        Estimator types that support exogenous variables (`supports_exog`).
    - `estimators_support_interval` : tuple
        Estimator types that support prediction intervals (`supports_interval`).

    """

    def _generate_estimator_ids(self) -> list[str]:
        """
        Generate unique estimator ids for a list of estimators.

        Handles duplicate estimator ids by appending a numeric suffix.
        
        Returns
        -------
        estimator_ids : list[str]
            List of unique ids for each estimator.
        
        """

        estimator_ids = []
        id_counts = {}
        for est in self.estimators:
            base_id = f"{type(est).__module__.split('.')[0]}.{type(est).__name__}"

            # Track occurrences and add suffix for duplicates
            if base_id in id_counts:
                id_counts[base_id] += 1
                unique_id = f"{base_id}_{id_counts[base_id]}"
            else:
                id_counts[base_id] = 1
                unique_id = base_id

            estimator_ids.append(unique_id)

        return estimator_ids

    def _build_estimators_repr_html(
        self, estimator_params: list[str]
    ) -> tuple[str, str]:
        """
        Build the HTML blocks for the estimators list and their parameters used
        in the `_repr_html_` method.

        Parameters
        ----------
        estimator_params : list
            Formatted parameters of each estimator as returned by
            `_preprocess_repr`.

        Returns
        -------
        estimators_html : str
            HTML block listing each estimator id and name.
        params_html : str
            HTML block listing the parameters of each estimator.

        """

        estimators_html = "<ul>"
        for est_id, est_name in zip(self.estimator_ids, self.estimator_names_):
            if est_name is not None:
                estimators_html += f"<li>{est_id}: {est_name}</li>"
            else:
                estimators_html += f"<li>{est_id}</li>"
        estimators_html += "</ul>"

        if len(estimator_params) == 1:
            params_html = f"<ul><li>{estimator_params[0]}</li></ul>"
        else:
            params_html = "<ul>"
            for param in estimator_params:
                params_html += f"<li>{param}</li>"
            params_html += "</ul>"

        return estimators_html, params_html

    def _check_select_fit_kwargs(
        self, fit_kwargs: dict[str, object] | None = None
    ) -> dict[str, dict[str, object]]:
        """
        Select, for each estimator, the keyword arguments accepted by its `fit`
        method. The same `fit_kwargs` provided by the user is validated against
        every estimator, since each one may expose a different `fit` signature.

        Parameters
        ----------
        fit_kwargs : dict, default None
            Dictionary with the arguments to pass to the `fit` method of the
            estimators.

        Returns
        -------
        fit_kwargs : dict
            Dictionary with the selected arguments for each estimator keyed by
            estimator id.

        """

        return {
            est_id: check_select_fit_kwargs(estimator=est, fit_kwargs=fit_kwargs)
            for est_id, est in zip(self.estimator_ids, self.estimators)
        }

    def get_estimator(self, id: str) -> object:
        """
        Get a specific estimator by its id.
        
        Parameters
        ----------
        id : str
            The id of the estimator to retrieve.
        
        Returns
        -------
        estimator : object
            The requested estimator instance.
        
        """
        
        if id not in self.estimator_ids:
            raise KeyError(
                f"No estimator with id '{id}'. "
                f"Available estimators: {self.estimator_ids}"
            )
        
        idx = self.estimator_ids.index(id)

        return self.estimators_[idx]
    
    def get_estimator_ids(self) -> list[str]:
        """
        Get the ids of all estimators in the forecaster.
        
        Returns
        -------
        estimator_ids : list[str]
            List of estimator ids.
        
        """

        return self.estimator_ids
    
    def remove_estimators(self, ids: str | list[str]) -> None:
        """
        Remove one or more estimators by their ids.
        
        Parameters
        ----------
        ids : str, list[str]
            The ids of the estimators to remove.
        
        Returns
        -------
        None
        
        """

        if isinstance(ids, str):
            ids = [ids]
        
        missing_ids = [id for id in ids if id not in self.estimator_ids]
        if missing_ids:
            raise KeyError(
                f"No estimator(s) with id '{missing_ids}'. "
                f"Available estimators: {self.estimator_ids}"
            )
            
        for id in ids:
            idx = self.estimator_ids.index(id)
            del self.estimators[idx]
            del self.estimators_[idx]
            del self.estimator_ids[idx]
            del self.estimator_names_[idx]
            del self.estimator_types[idx]
            del self.estimator_params_[id]
            self.n_estimators -= 1

    def get_estimators_info(self) -> pd.DataFrame:
        """
        Get a summary DataFrame with information about all estimators in the 
        forecaster.
        
        Returns
        -------
        info : pandas DataFrame
            DataFrame with columns:

            - id: Unique identifier for each estimator.
            - name: Descriptive name (available after fitting).
            - type: Full qualified type string.
            - supports_exog: Whether the estimator supports exogenous variables.
            Only included if the forecaster defines `estimators_support_exog`.
            - supports_interval: Whether the estimator supports prediction
            intervals. Only included if the forecaster defines
            `estimators_support_interval`.
            - params: Dictionary of the estimator parameters.
        
        """

        info = {
            'id': self.estimator_ids,
            'name': self.estimator_names_,
            'type': self.estimator_types,
        }

        if hasattr(self, 'estimators_support_exog'):
            info['supports_exog'] = [
                est_type in self.estimators_support_exog
                for est_type in self.estimator_types
            ]
        if hasattr(self, 'estimators_support_interval'):
            info['supports_interval'] = [
                est_type in self.estimators_support_interval
                for est_type in self.estimator_types
            ]

        info['params'] = [
            str(self.estimator_params_[est_id]) for est_id in self.estimator_ids
        ]

        info = pd.DataFrame(info)

        return info
