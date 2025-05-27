import argparse
def get_parser()-> argparse.ArgumentParser:
    # Implementation of ID:134
    parser = argparse.ArgumentParser( description="RunDetails argument parser" )

    parser.add_argument( '--dataset_name', type=str, required=True, help='Name of the dataset' )
    parser.add_argument( '--model_size', type=str,choices = ['smallu','largeu'], required=True, help='size of the model' )
    parser.add_argument( '--scheduler_type', type=str,choices = ['one_cycle','cosine'], required=True, help='type of scheduler' )
    parser.add_argument( '--environment', type=str, choices=['laptop', 'cluster', 'bwcluster'], required=True,
                         help='Execution environment (laptop or cluster)' )
    parser.add_argument( '--run_notes', type=str, required=True,
                         help='Documentation of the run' )
    parser.add_argument( '--checkpoint', type=str, required=False )
    parser.add_argument('--batch_size', type=int, required=False, choices=[2,4,8,16,32,64,128])
    parser.add_argument('--num_epochs', type=int, required=False, choices=[10,30,50,100,200])
    parser.add_argument('--ema', type=float, required=False, choices=[0.9,0.99,0.999,0.9999])
    parser.add_argument('--lr', type=float, required=False, choices=[0.001,0.0001,0.00001,0.0005])
    parser.add_argument('--weight_decay', type=float, required=False, choices=[0.01,0.001,0.00005,0.005])




    return parser


