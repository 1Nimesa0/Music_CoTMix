import os
import argparse
import warnings
from trainer import cross_domain_trainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser(description="CoTMix Cross-Domain Adaptation Engine")

# ======== Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='cross_environment_music',  type=str, help='Name of your experiment')
parser.add_argument('--run_description',        default='MusicGenre_DinoV3',        type=str, help='Name of your run')
parser.add_argument('--sweep_project',          default='cotmix_music_sweep',       type=str, help='Name of your sweep project')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='CoTMix',      type=str, help='DA Method: CoTMix')

# ========= Select the DATASET ==============
parser.add_argument('--dataset',                default='MusicGenre',  type=str, help='Dataset of choice: MusicGenre, HHAR, WISDM, EEG, HAR')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='DinoV3',      type=str, help='Backbone of choice: DinoV3, CNN')

# ========= Experiment settings ===============
parser.add_argument('--data_path',              default=r'./data/',    type=str,   help='Path containing dataset')
parser.add_argument('--num_runs',               default=1,             type=int,   help='Number of consecutive runs with different seeds')
parser.add_argument('--device',                 default='cuda:0',      type=str,   help='cpu or cuda:0')
parser.add_argument('--num_epochs',             default=None,          type=int,   help='Optional override for number of training epochs')
parser.add_argument('--batch_size',             default=None,          type=int,   help='Optional override for batch size')
parser.add_argument('--is_sweep',               default=False,         type=bool,  help='Single run or wandb sweep')
parser.add_argument('--num_sweeps',             default=100,           type=str,   help='Number of sweep runs')

args = parser.parse_args()

if __name__ == "__main__":
    trainer = cross_domain_trainer(args)

    if args.is_sweep:
        trainer.sweep()
    else:
        trainer.train()
