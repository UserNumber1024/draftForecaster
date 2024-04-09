from lib.predictors import Predictor
from lib.preparer import MOEXtoXY
import warnings
import lib.constants as c
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------
# SETUP PARAMS OF EXPERIMENT HERE----------------------------------------------
# -----------------------------------------------------------------------------
Y_type = c.clf
under_sampling = c.KNN
model_type = c.forest
search_type = c.bayes_search
source_tickers = c.file
source_XY = c.calc
draw_xy = False
draw_pca = False
draw_tSNE = False
train_model = True
print_metrics = True
draw_metrics = True
draw_model = False
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
preparer = MOEXtoXY(debug=c.debug)
X, Y, prices = preparer.prepare_XY(source_tickers=source_tickers,
                                   source_XY=source_XY,
                                   length=c.length,
                                   profit_margin=c.profit_margin,
                                   Y_type=Y_type,
                                   under_sampling = under_sampling)
if draw_xy:
    preparer.draw_X_Y()
if draw_pca:
    preparer.draw_PCA()
if draw_tSNE: 
    preparer.draw_tSNE(n_jobs=-1, n_iter=500)
if train_model:
    predictor = Predictor(X=X, Y=Y, Y_type=Y_type, debug=c.debug)
    best_tree = predictor.tree_search(search_type=search_type,
                                      model_type=model_type,
                                      n_jobs=c.n_jobs, random_n_iter=c.random_n_iter)
    if print_metrics:
        predictor.print_metrics()
    if draw_metrics:
        predictor.draw_metrics()
    if draw_model:
        predictor.draw_model()
