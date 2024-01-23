import os
import torch
import logging
import numpy as np
from core.data import Data_Loader
from core.model import Model_Loader
from core.model import Checkpoint_Loader
from core.trainer import Untrain_Loader
from core.args import Untrain_Parser

import time

def run(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Model_Loader(args)(name=args.model,
                               num_classes=args.num_classes)
    model = Checkpoint_Loader(args, model)
    trainloaders, testloaders = Data_Loader(args)

    # Logger setup
    run_name = args.method+"_"+args.unlearn_type+"_"+time.strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs("./logging", exist_ok=True)
    logging.basicConfig(filename=os.path.join("./logging", run_name),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger("Args")
    for _arg, _val in sorted(vars(args).items()):
        logger.info("%s: %r", _arg, _val)
    logger.info("Unlearning instance initiated")
    logger = logging.getLogger("Unlearn")

    if args.do_save is True or args.last_save:
        save_path = os.path.join(args.save_path, args.dataset, args.model, run_name)
        os.makedirs(save_path, exist_ok=True)
        args.save_path = save_path
        logger.info(f"Save model at {save_path}")

    trainer = Untrain_Loader(args)(model, trainloaders, testloaders, args)

    for e in range(args.unlearn_epoch):
        unlearn_stats = trainer.fit(e)
        ts_acc, ts_conf = trainer.evaluate()
        tr_acc, tr_conf = trainer._evaluate(trainloaders[1])

        report = dict()
        report.update(unlearn_stats)

        if len(ts_acc) > 1:
            report["test_unlearn_acc"] = ts_acc[0]
            report["test_retain_acc"] = ts_acc[1]

        else:
            report["test_acc"] = ts_acc[0]

        report["train_acc"] = tr_acc

        _report = f"Epoch: {e}, "
        for _key, val in report.items():
            _report += f"{_key}: {val}, "
        logger.info(_report)
        
        if args.do_save is True:
            logger.info(f"Epoch {e}, save model")
            os.makedirs(os.path.join(save_path, str(e)), exist_ok=True)
            torch.save({
                "state_dict": trainer.model.state_dict(),
                "dataset": args.dataset,
                "test_acc": ts_acc[0],
                "model": args.model
                }, os.path.join(save_path, str(e), "model.pt"))
            
        # Check termination condition
        if args.method == "contrastive" and args.unlearn_type == "random_sample":
            if report["unlearn_acc"] <= report["test_acc"]:
                logger.info(f"Stopping condition has been satisfied at epoch {e}")
                break
        if args.method == "contrastive" and args.unlearn_type == "single_class":
            if report["test_unlearn_acc"] <= (1.0/args.num_classes):
                logger.info(f"Stopping condition has been satisfied at epoch {e}")
                break

    if args.last_save is True:
        logger.info("Save unlearned model satisfying termination condition")
        os.makedirs(os.path.join(save_path, "termination"), exist_ok=True)
        torch.save({
            "state_dict": trainer.model.state_dict(),
            "dataset": args.dataset,
            "test_acc": ts_acc[0],
            "model": args.model
            }, os.path.join(save_path, "termination", "model.pt"))

    logger.info("Terminated")


if __name__ == "__main__":
    args = Untrain_Parser().parse_args()
    run(args)

