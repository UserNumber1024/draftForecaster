from lib.predictors import Predictor
from lib.preparer import MOEXtoXY
import warnings
import lib.constants as c
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------
# SETUP PARAMS OF EXPERIMENT HERE----------------------------------------------
# -----------------------------------------------------------------------------
source_tickers = c.file         # file, moex
source_XY = c.file              # file, calc
store_tickers2file = False      # set false after downloading actual data
store_XY2file = False           # set false after preparing actual X and Y
draw_xy = False
train_model = True
print_metrics = True
draw_metrics = False
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
preparer = MOEXtoXY()
X, Y, prices = preparer.prepare_XY(
    store_tickers2file=store_tickers2file,
    store_XY2file=store_XY2file,
    source_tickers=source_tickers,
    source_XY=source_XY,
    profit_margin=c.profit_margin)
if draw_xy:
    preparer.draw_X_Y()

if train_model:
    # 'profit_1'
    # 'mean_delta_1'
    predictor = Predictor(X=X, Y=Y['profit_1'], Y_type=c.clf, debug=c.debug)
    best_tree = predictor.tree_search(n_jobs=c.n_jobs,
                                      n_trials=c.n_trials)
    if print_metrics:
        predictor.print_metrics()
    if draw_metrics:
        predictor.draw_metrics()

