from core.trainer.learn.learn import NormalTrainer
from core.trainer.unlearn.unlearn import UnlearnTrainer
from core.trainer.unlearn.retrain import Retrain


def Train_Loader(args):
    
    return NormalTrainer

def Untrain_Loader(args):

    if args.method == "retrain":
        return Retrain

    elif args.method == "contrastive":
        return UnlearnTrainer
    
    else:
        raise ValueError(f"Unknown unlearning method : {args.method}")
