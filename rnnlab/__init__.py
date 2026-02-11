"""rnnlab: minimalistyczna biblioteka do mini‑projektów A/C/D.

Założenia projektowe:
- wszystko działa na CPU (bez CUDA) w rozsądnym czasie,
- dane są syntetyczne (żeby izolować zjawiska, a nie walczyć z datasetem),
- każdy eksperyment zapisuje wyniki do CSV (łatwo scalać pracę wielu osób).
"""

from .utils import set_seed, get_device, count_runs, cpu_friendly, now_tag
from .data import TaskSpec, make_fixed_testset, sample_batch
from .models import ModelSpec, SeqClassifier
from .train import TrainConfig, train_run, evaluate
from .plotting import plot_heatmap, plot_lines_acc, plot_acc_over_steps

from .experiment_a import sweep_frontier, summarize_frontier
#from .experiment_c import sweep_curriculum
#from .experiment_d import sweep_optimizers
