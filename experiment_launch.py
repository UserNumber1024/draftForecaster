from lib.predictors import Predictor
from lib.preparer import MOEXtoXY
import warnings

warnings.filterwarnings("ignore")

# SETUP PARAMS OF EXPERIMENT HERE----------------------------------------------
# -----------------------------------------------------------------------------
Y_type = 'clf'
model_type = 'tree'
search_type = 'bayes_search'
source = 'file'
length = 1
profit_margin = 0.01
debug = 2
n_jobs = 1
random_n_iter = 20
draw_xy = True
draw_pca = True
draw_tSNE = True
train_model = True
print_metrics = True
draw_metrics = True
draw_model = False
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

preparer = MOEXtoXY()
X, Y, prices = preparer.prepare_XY(source=source,
                                   length=length,
                                   profit_margin=profit_margin,
                                   Y_type=Y_type)
if draw_xy:
    preparer.draw_X_Y()
if draw_pca:
    preparer.draw_PCA()
if draw_tSNE:
    preparer.draw_tSNE(debug=debug, n_jobs=-1, n_iter=500)
if train_model:
    predictor = Predictor(X=X, Y=Y, Y_type=Y_type, debug=debug)
    best_tree = predictor.tree_search(search_type=search_type,
                                      model_type=model_type,
                                      n_jobs=n_jobs, random_n_iter=random_n_iter)
    if print_metrics:
        predictor.print_metrics()
    if draw_metrics:
        predictor.draw_metrics()
    if draw_model:
        predictor.draw_model()
