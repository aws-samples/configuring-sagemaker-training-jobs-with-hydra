import logging
from pathlib import Path

import hydra
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from mypackage import config
from mypackage.evaluation import save_comparison_plot_ordered, score_estimator

config_path = Path(config.__file__).parent
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg):
    """Main harness to train the model."""
    logging.info(f"Loading data set from {cfg.data.train_data_path}.")
    df = pd.read_csv(cfg.data.train_data_path)

    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    logging.info(
        f"Train_test_split with test size {cfg.data.test_size} and random state {cfg.data.random_state_split}."
    )
    df_train, df_test = train_test_split(
        df, test_size=cfg.data.test_size, random_state=cfg.data.random_state_split
    )

    tree_preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
            ("numeric", "passthrough", ["VehAge", "DrivAge", "BonusMalus", "Density"]),
        ],
        remainder="drop",
    )
    poisson_gbrt = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="poisson", max_leaf_nodes=cfg.model.max_leaf_nodes
                ),
            ),
        ]
    )
    poisson_gbrt.fit(
        df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
    )
    logging.info("Training completed.")

    score_estimator(estimator=poisson_gbrt, df_test=df_test)
    save_comparison_plot_ordered(
        estimator=poisson_gbrt,
        df_test=df_test,
        path=Path(cfg.evaluation.base_dir_path) / "comparison.png",
    )
    logging.info(f"Evaluation information saved to {cfg.evaluation.base_dir_path}.")


if __name__ == "__main__":
    main()
