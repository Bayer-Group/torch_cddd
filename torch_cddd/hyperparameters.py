DEFAULT_TRAIN_PATH_CSV = "data/train.csv"
DEFAULT_EVAL_PATH_CSV = ""

def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    parser.add_argument('-d', '--device', default="cpu", type=str)
    parser.add_argument('-e', '--num_epochs', default=50, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--num_eval_samples", default=2000, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN_PATH_CSV, type=str)
    parser.add_argument("--eval_csv", default=DEFAULT_EVAL_PATH_CSV, type=str)
    #parser.add_argument("-s", "--save_dir", type=str, required=True)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--tf_ratio", default=1.0, type=float)
    parser.add_argument("--tf_step_freq", default=100000, type=int)
    parser.add_argument("--tf_step_rate", default=0.9, type=float)
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.set_defaults(multi_gpu=False)
    parser.add_argument('--test', dest='test_run', action='store_true')
    parser.set_defaults(test_run=False)
    return parser
