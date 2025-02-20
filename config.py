from absl import flags
from typing import Optional
import argparse
import sys
from datetime import datetime
def get_parser()-> argparse.ArgumentParser:
    # Implementation of ID:134
    parser = argparse.ArgumentParser( description="RunDetails argument parser" )

    parser.add_argument( '--dataset_name', type=str, required=True, help='Name of the dataset' )
    parser.add_argument( '--model_id', type=str, required=True, help='Name of the model' )
    parser.add_argument( '--version', type=str, required=True, help='Model version (plain or modified)' )
    parser.add_argument( '--environment', type=str, choices=['laptop', 'cluster', 'bwcluster'], required=True,
                         help='Execution environment (laptop or cluster)' )
    parser.add_argument( '--train_state', type=str, choices=['T', 'NT'], required=True,
                         help='Is training wanted? (T(raining) / N(o)T(raining)' )
    parser.add_argument( '--date', type=str, default=datetime.now().strftime( '%Y-%m-%d' ),
                         help='Current date (default: today)' )
    parser.add_argument( '--device', type=str, required=True, help='Device to be used (e.g., cuda or cpu)' )
    parser.add_argument( '--task', type=str, choices=['classification', 'transcribe', 'joint'], required=True,
                         help='Task type (classification, transcription, or joint)' )
    parser.add_argument( '--developer_mode', type=str, choices=['Y', 'N'], required=True,
                         help='Developer mode (yes for small datasets, no for full training)' )
    parser.add_argument( '--augmentation', type=str, choices=['Y', 'N'], required=True,
                         help='Use synthetic noise augmentation can be Y(es) or N(o)' )
    parser.add_argument( '--additional_tokens', type=str, choices=['Y', 'N'], required=True,
                         help='Add additonal tokens of the dataset to the network can be Y(es) or N(o)' )
    parser.add_argument( '--run_notes', type=str, required=True,
                         help='Documentation of the run' )
    parser.add_argument('--dataset_evaluation_part', type=str, choices=['dev','eval'],required=False)
    parser.add_argument( '--oversampling_clean_data', type=int, choices=[1,2,3,4,5,6,7,8,9,10], required= True )
    parser.add_argument( '--checkpoint', type=str, required=False )
    parser.add_argument( '--data_portion', type=str, choices=["clean-only", "far-only", "all"], required=True )
    parser.add_argument( '--beamforming', type=str, choices=["Y","N"], required=False )
    parser.add_argument('--SWAD', type=bool, required=False)
    parser.add_argument('--diffusion', type=str, required=False, choices=['Y','N'])
    return parser

parser = get_parser()
args = parser.parse_args()
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', None, 'Name of the dataset', required=True)
flags.DEFINE_string('model_id', None, 'Name of the model', required=True)
flags.DEFINE_string('version', None, 'Model version (plain or modified)', required=True)
flags.DEFINE_enum('environment', None, ['laptop', 'cluster', 'bwcluster'], 'Execution environment', required=True)
flags.DEFINE_enum('train_state', None, ['T', 'NT'], 'Is training wanted? (T(raining) / N(o)T(raining)', required=True)
flags.DEFINE_string('date', datetime.now().strftime('%Y-%m-%d'), 'Current date (default: today)')
flags.DEFINE_string('device', None, 'Device to be used (e.g., cuda or cpu)', required=True)
flags.DEFINE_enum('task', None, ['classification', 'transcribe', 'joint'], 'Task type', required=True)
flags.DEFINE_enum('developer_mode', None, ['Y', 'N'], 'Developer mode (yes for small datasets, no for full training)', required=True)
flags.DEFINE_enum('augmentation', None, ['Y', 'N'], 'Use synthetic noise augmentation', required=True)
flags.DEFINE_enum('additional_tokens', None, ['Y', 'N'], 'Add additional tokens to the network', required=True)
flags.DEFINE_string('run_notes', None, 'Documentation of the run', required=True)
flags.DEFINE_enum('dataset_evaluation_part', None, ['dev', 'eval'], 'Dataset evaluation part', required=False)
flags.DEFINE_integer('oversampling_clean_data', None, 'Oversampling clean data (1-10)', lower_bound=1, upper_bound=10, required=True)
flags.DEFINE_string('checkpoint', None, 'Checkpoint path', required=False)
flags.DEFINE_enum('data_portion', None, ['clean-only', 'far-only', 'all'], 'Data portion', required=True)
flags.DEFINE_enum('beamforming', None, ['Y', 'N'], 'Beamforming enabled', required=False)
flags.DEFINE_boolean('SWAD', False, 'SWAD flag', required=False)
flags.DEFINE_enum('diffusion', None, ['Y', 'N'], 'Diffusion enabled', required=False)

