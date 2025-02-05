# -----------------------------------------------------------------------------
# CONSTANTS -------------------------------------------------------------------
# -----------------------------------------------------------------------------
debug = 1  # verbosity level
# ---------------------------------------------------------
rand = 'rand'
file = 'file'
moex = 'moex'
calc = 'calc'
clf = 'clf'
regr = 'regr'
# ---------------------------------------------------------
n_jobs = 2  # jobs for search hyperparams (-1 use all cores)
n_trials = 4  # trials for optuna
seed = 0.2  # part of the data for testing
timeout = 300 # stop optuna study after the given number of second(s) 
task_type = 'CPU' # processor used for Catboost training (CPU / GPU)
gc_after_trial = True # garbage collection after each trial
# ---------------------------------------------------------
n_days = 60  # days for normalization by mean
profit_margin = 0.01  # price increase meaning "profit"
up_quant = 0.95  # used to normalize/scale indicators
low_quant = 0.05  # used to normalize/scale indicators
start = "2015-01-01"
end = "2024-12-01"
# ---------------------------------------------------------
# values (0.01, 0.02 not used and make no sense)
moex_ticker_ids = {
    'SBER': 0.01,
    'SBERP': 0.02,
    'LKOH': 0.03,
    'GAZP': 0.04,
    'TATN': 0.05,
    'TATNP': 0.06,
    'SNGS': 0.07,
    'SNGSP': 0.08,
    'YDEX': 0.09,
    'GMKN': 0.10,
    # 'T': 0.11,
    'NVTK': 0.12,
    'PLZL': 0.13,
    'ROSN': 0.14,
    'CHMF': 0.15,
    'IRAO': 0.16,
    'NLMK': 0.17,
    'OZON': 0.18,
    'MOEX': 0.19,
    'RUAL': 0.20,
    'SIBN': 0.21,
    'MRKV': 0.22,
    'AFLT': 0.23,
    'MTSS': 0.23,
    'POSI': 0.24,
    'POLY': 0.25,
    'PHOR': 0.26
}
# ----------------------------------------------------------
file_folder = 'data/'
graph_folder = 'graph/'
model_folder = 'model/'
