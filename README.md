# LC-beating
The training part and inference part of LC-beating mechanism



## training
all the training datasets are saved in the .h5 file.
you can use your own dataloader to load data for training and evaluation.

```
python train.py --network='lc42_tcn --mark='training' --eval_mode='online' --peak_type='simple'
```


## inference
inference part not yet implemented, source code for reference only.