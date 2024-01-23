from argparse import ArgumentParser


def Train_Parser():
    parser = ArgumentParser()

    # 1. General setting
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--do_save", default=False, action='store_true', help="Save the model for every epoch")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--save_path",  type=str, help="Path to save the model")
    parser.add_argument("--datapath",   type=str,   default="./dataset", help="Path to download datasets (CIFAR10 and SVHN)")
    parser.add_argument("--dataset",    type=str,   default="cifar_10", choices=["cifar_10", "svhn"])
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--cropsize",   type=int,   default=32, help="Size of the input image")
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--device",     type=str,   default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--momentum",   type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay",   type=float, default=0.1)
    parser.add_argument("--w_decay",    type=float, default=1e-4)


    return parser


def Untrain_Parser():
    parser = ArgumentParser()

    # 1. General setting
    parser.add_argument("--method", type=str, default="contrastive", choices=["contrastive", "retrain"])
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--load_path",  type=str, help="Path to load a trained model")
    parser.add_argument("--do_save", default=False, action='store_true', help="Save the unlearned model for every epoch")
    parser.add_argument("--last_save", default=False, action='store_true', help="Save the unlearned model at the last epoch")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--save_path",  type=str, help="Path to save the unlearned model")
    parser.add_argument("--datapath",   type=str,   default="./dataset")
    parser.add_argument("--dataset",    type=str,   default="cifar_10")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--cropsize",   type=int,   default=32)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--device",     type=str,   default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unlearn_type", type=str, default="random_sample", choices=["random_sample", "single_class"], help="Unlearning task")
    args = parser.parse_known_args()[0]
    

    # 2. Unlearning setting
    # Specify unlearning objective (class vs sample unlearn)

    if args.unlearn_type == "random_sample":
        parser.add_argument("--num_unlearn", type=float, default=500, help="Number of samples to unlearn")
    
    if args.unlearn_type == "single_class":
        parser.add_argument("--unlearn_class", type=int, default=5)
    
    
    # 3. Algorithm specific settings

    ###### Proposed Method ######
    if args.method == "contrastive":

        parser.add_argument("--temp",       type=float, default=0.7, help="temperature parameter")
        parser.add_argument("--loss_ver",   type=int,   default=2, help="Loss versions")
        parser.add_argument("--regularize", type=bool,  default=True)
        parser.add_argument("--retain_sampling_freq", type=int, default=4, help="Sampling frequency of retaining data")
        parser.add_argument("--CT_ratio", type=float, default=3.0, help="Hyperparameter to determine influence of contrastive unlearning loss (equivalent to \lambda_\{UL\})")
        parser.add_argument("--CE_ratio", type=float, default=1.0, help="Hyperparameter to determine influence of cross-entropy loss (equivalent to \lambda_\{CE\})")
        parser.add_argument("--unlearn_epoch", type=int, default=3, help="Number of iteration over forgetting set") 
        parser.add_argument("--lr",         type=float, default=1e-3)
        parser.add_argument("--lr_decay",   type=float, default=1e-3)
        parser.add_argument("--w_decay",    type=float, default=1e-3)
        parser.add_argument("--momentum",   type=float, default=0.9)

    ###### Baselines ######
    elif args.method == "retrain":
        
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--lr_decay",   type=float, default=0.1)
        parser.add_argument("--w_decay",    type=float, default=1e-4)
        parser.add_argument("--momentum",   type=float, default=0.9)
        parser.add_argument("--unlearn_epoch", type=int, default=100, help="Number of iteration over forgetting set")
    
    return parser