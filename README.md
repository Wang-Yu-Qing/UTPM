# UTPM (under developing)
Code for paper: [Learning to Build User-tag Profile in Recommendation System](https://dl.acm.org/doi/abs/10.1145/3340531.3412719)

## Dataset
Download link: [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
Create data folder, unzip and move the dataset into the folder

## Run the model
Run with default config

```
python main.py
```

If everything is properly set, you should see outputs like the following:

```
epoch: 000 | current_step: 00000 | current_batch_loss: 6.6119 | epoch_avg_loss: 6.6119 | step_time: 0.35378
epoch: 000 | current_step: 00100 | current_batch_loss: 3.0183 | epoch_avg_loss: 4.4048 | step_time: 0.01175
epoch: 000 | current_step: 00200 | current_batch_loss: 5.4969 | epoch_avg_loss: 3.7021 | step_time: 0.01200
epoch: 000 | current_step: 00300 | current_batch_loss: 4.1210 | epoch_avg_loss: 3.3223 | step_time: 0.01197
epoch: 000 | current_step: 00400 | current_batch_loss: 1.5884 | epoch_avg_loss: 3.0219 | step_time: 0.01407
epoch: 000 | current_step: 00500 | current_batch_loss: 0.5027 | epoch_avg_loss: 2.7848 | step_time: 0.01389
epoch: 000 | current_step: 00600 | current_batch_loss: 3.1471 | epoch_avg_loss: 2.5947 | step_time: 0.00737
epoch: 000 | current_step: 00700 | current_batch_loss: 0.9000 | epoch_avg_loss: 2.4497 | step_time: 0.00798
epoch: 000 | current_step: 00800 | current_batch_loss: 0.6784 | epoch_avg_loss: 2.3137 | step_time: 0.00795
epoch: 000 | current_step: 00900 | current_batch_loss: 1.1106 | epoch_avg_loss: 2.1893 | step_time: 0.00767
epoch: 000 | current_step: 01000 | current_batch_loss: 0.7049 | epoch_avg_loss: 2.0353 | step_time: 0.00759
......
```

To change config, pass arguments when launch main.py, check utils.py for arguments details.

For example, run the model with cpu, batch size of 64 and max samples per user being 20:

```
python main.py --gpu False --batch_size 64 --max_user_samples 20
```

## Evaluation

