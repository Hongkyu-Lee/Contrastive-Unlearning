# Contrastive Unlearning: A Contrastive Approach to Machine Unlearning

This is an official implementation of the paper [Contrastive unlearning: A Contrastive Approach to Machine Unlearning]().

**Envrionment Details**  
* python=3.10.12  
* pytorch=2.0.1  
* torchvision=0.15.2  

## How to run

### 0. Train a model

Train a model using ```train.py``` . Example code is as follows.
```
python train.py --model resnet18 \
                --dataset cifar_10 \
                --epochs 2 \
                --lr 0.1 \
                --save_path "/path/to/save/the/model/"

```

### 1. Run Contrastive Unlearning
Run ```unlearn.py``` for unlearning a model. You should provide ```load_path``` argument to load a trained model. Here are example codes for unleanring single class and random sample task.

```
python unlearn.py   --method contrastive \
                    --model resnet18 \
                    --unlearn_type single_class \
                    --unlearn_class 5 \
                    --batch_size 64 \
                    --unlearn_epoch 1 \
                    --load_path "/path/to/a/trained/model.pt" \
                    --last_save \
                    --save_path "/path/to/save/the/unlearned/model/"
```

```
python unlearn.py   --method contrastive \
                    --model resnet18 \
                    --unlearn_type random_sample \
                    --num_unlearn 500 \
                    --unlearn_epoch 100 \
                    --retain_sampling_freq 4 \
                    --batch_size 128 \
                    --unlearn_epoch 30 \
                    --load_path "/path/to/a/trained/model.pt" \
                    --last_save \
                    --save_path "/path/to/save/the/unlearned/model/"
```

### 2. Run Baseline (Retrain)
To run get a retrained model, change unlearning method to ```retrain```, or replace it into ```--method retrain```. Example code is as follows

```
python unlearn.py   --method retrain \
                    --model resnet18 \
                    --dataset cifar_10 \
                    --unlearn_type single_class \
                    --unlearn_class 5 \
                    --batch_size 256 \
                    --unlearn_epoch 2 \
                    --last_save \
                    --save_path "/path/to/save/the/unlearned/model/"
                    
```
