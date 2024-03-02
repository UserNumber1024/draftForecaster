from predictors import TreePredictor
from preparer import MOEXtoXY
import warnings
warnings.filterwarnings("ignore")

# SETUP PARAMS OF EXPERIMENT HERE----------------------------------------------
# -----------------------------------------------------------------------------
Y_type = 'clf'
model_type = 'tree'
search_type = 'rand'
source = 'file'
length = 1
debug = 2
n_jobs = 1
random_n_iter = 15
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

preparer = MOEXtoXY()

X, Y, prices = preparer.prepare_XY(source=source,
                                   length=length,
                                   Y_type=Y_type)

preparer.draw_X_Y()
# preparer.draw_PCA()

# preparer.draw_tSNE(debug=debug, n_jobs=-1, n_iter=500)

predictor = TreePredictor(X=X, Y=Y, Y_type=Y_type, debug=debug)
best_tree = predictor.tree_search(search_type=search_type,
                                  model_type=model_type,
                                  n_jobs=n_jobs, random_n_iter=random_n_iter)

predictor.print_metrics()
predictor.draw_metrics()
# pred.draw_model()
