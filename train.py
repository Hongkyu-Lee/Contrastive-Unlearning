import os
import torch
import numpy as np
from core.data.datasets import DATASETS
from core.model import Model_Loader
from core.trainer import Train_Loader
from core.args import Train_Parser

import time
import logging

def run(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.model+"_"+args.dataset+"_"+str(args.epochs)+"_"+ time.strftime("%Y_%m_%d_%H_%M_%S")
    model = Model_Loader(args)(name=args.model,
                               num_classes=args.num_classes)

    train_data, test_data = DATASETS[args.dataset](args)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                               shuffle=None, num_workers=args.num_workers, pin_memory=True)
    
    save_path = os.path.join(args.save_path, run_name)
    os.makedirs(save_path, exist_ok=True)
    args.save_path = save_path

    os.makedirs("./logging", exist_ok=True)
    logging.basicConfig(filename=os.path.join("./logging", run_name),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger("Args")
    for _arg, _val in sorted(vars(args).items()):
        logger.info("%s: %r", _arg, _val)
    logger.info("Train instance initiated")
    logger = logging.getLogger("Train")
    logger.info(f"Save trained model at: {save_path}")


    trainer = Train_Loader(args)(model, train_loader, (test_loader,), args)
    
    best_test_acc = 0.0
    best_state_dict = model.state_dict()

    for e in range(args.epochs):
        loss, train_acc = trainer.fit(e)
        test_acc, conf = trainer.evaluate()

        if (test_acc > best_test_acc):
            best_state_dict = trainer.model.state_dict()
            best_test_acc = test_acc

        logger.info(f"Epoch {e}, Train Loss: {loss}, Train Acc: {train_acc}, Test Acc: {test_acc}")
    
    # save
    logger.info("Training finished")
    logger = logging.getLogger("Save")

    os.makedirs(os.path.join(save_path, "best"), exist_ok=True)
    torch.save({
        "state_dict": best_state_dict,
        "dataset": args.dataset,
        "test_acc": best_test_acc,
        }, os.path.join(save_path, "best", "model.pt"))
    
    logger.info(f"Model with test accuracy {best_test_acc} has beens saved.")
    logger.finish("Terminate")

if __name__ == "__main__":
    args = Train_Parser().parse_args()
    print(args)
    run(args)

