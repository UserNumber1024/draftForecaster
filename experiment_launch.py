import warnings
import sys
from lib.predictorCatBoost import ModelCatBoost
from lib.preparer import MOEXtoXY
import lib.constants as c
warnings.filterwarnings("ignore")


class ExperimentConfig:
    """Setup experiment parameters."""

    def __init__(self):
        self.source_tickers = c.file         # file, moex
        self.source_XY = c.calc              # file, calc
        self.store_tickers2file = False      # false after downloading data
        self.store_XY2file = True           # false after preparing X and Y
        self.draw_xy = True
        self.train_model = True
        self.print_metrics = True
        self.draw_metrics = True
        self.export_model = True


def log_output(log_file: str):
    """Redirect print to a file."""

    class LogOutput:
        def __enter__(self):
            self.original_stdout = sys.stdout
            sys.stdout = open(log_file, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self.original_stdout

    return LogOutput()


def run_experiment(config: ExperimentConfig):
    """Run models learning."""
    preparer = MOEXtoXY()
    X, Y, prices = preparer.prepare_XY(
        store_tickers2file=config.store_tickers2file,
        store_XY2file=config.store_XY2file,
        source_tickers=config.source_tickers,
        source_XY=config.source_XY,
        profit_margin=c.profit_margin
    )

    if config.draw_xy:
        preparer.draw_X_Y()

    # for period in [1, 2, 3, 4, 5]:
    for period in [1]:
        # for model_type in [c.clf]:
        for model_type in [c.regr, c.clf]:
            y_name = "profit_" if model_type == c.clf else "mean_delta_"
            y_name = f"{y_name}{period}"

            name = f"{model_type}_catbst_{period}"

            if config.train_model:
                model = ModelCatBoost(
                    X=X, Y=Y[y_name], Y_type=model_type, name=name)
                model.optimize_params()
                model.fit()

                if config.print_metrics:
                    model.print_best_params()
                    model.print_metrics()

                if config.draw_metrics:
                    model.draw_metrics()

                if config.export_model:
                    model.export_model()

    return X, Y, prices


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = ExperimentConfig()

    with log_output('log/log.txt'):
        X, Y, prices = run_experiment(config)

    print("Learning completed.")
