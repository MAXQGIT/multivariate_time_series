import argparse


def Argspare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PatchTST', required=False)
    parser.add_argument('--data_path', type=str, default='datasets/raw_data', required=False)
    parser.add_argument('--history_len', type=int, default=168, required=False)
    parser.add_argument('--predict_len', type=int, default=24, required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    parser.add_argument('--train_rate', type=float, default=0.8, required=False)
    parser.add_argument('--val_rate', type=float, default=0.1, required=False)
    parser.add_argument('--lr', type=float, default=0.0001, required=False)
    parser.add_argument('--epochs', type=int, default=2000, required=False)
    parser.add_argument('--device', type=str, default='cuda', required=False)
    parser.add_argument('--kernel_size', type=int, default=3, required=False)
    parser.add_argument('--input_size', type=int, default=11, required=False)
    parser.add_argument('--hidden_size', type=int, default=128, required=False)
    parser.add_argument('--step_size', type=int, default=10, required=False)
    parser.add_argument('--gamma', type=float, default=0.7, required=False)
    parser.add_argument('--model_save_path', type=str, default='model/deeplearning_model', required=False)
    parser.add_argument('--patience', type=int, default=10, required=False)
    parser.add_argument('--num_workers', type=int, default=0, required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.1, required=False)
    parser.add_argument('--is_scaler', type=int, default=0, required=False, help='Data standard')
    # PatchTST
    parser.add_argument('--patch_len', type=int, default=16, required=False)
    parser.add_argument('--stride', type=int, default=8, required=False)
    parser.add_argument('--n_heads', type=int, default=8, required=False)
    parser.add_argument('--d_ff', type=int, default=1024, required=False)
    parser.add_argument('--n_layers', type=int, default=2, required=False)
    # SegRNN
    parser.add_argument('--affine', type=int, default=1, required=False, help='Please select 1 or 0')
    parser.add_argument('--mode', type=str, default='norm', required=False, help='Please select "norm" or "denorm"')
    parser.add_argument('--seg_len', type=int, default=24, required=False, help='segment length')
    parser.add_argument('--dec_way', type=str, default='rmf', required=False,
                        help='decode way,please select "rmf" "pmf" ')
    args = parser.parse_args()
    return args
