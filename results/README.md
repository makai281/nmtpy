|   id |   src vocab |   trg vocab |   train |   valid |   emb_dim |   rnn_dim | BLEU            | METEOR     | loss        |
|-----:|------------:|------------:|--------:|--------:|----------:|----------:|:----------------|:-----------|:------------|
|   10 |       11249 |       18724 |   29000 |    1014 |       620 |      1000 | **35.480 (17)** | 0.540 (17) | 28.125 (17) |
|   12 |       11249 |       19217 |   29000 |    1014 |       620 |      1000 | 35.090 (17)     | 0.534 (17) | 27.984 (17) |
|    9 |       11247 |       18723 |   28998 |    1014 |       620 |      1000 | 35.020 (17)     | 0.539 (17) | 27.967 (17) |
|   11 |       11249 |       18724 |   29000 |    1014 |       620 |      1200 | 34.420 (17)     | 0.541 (17) | 28.151 (17) |
|    1 |       11156 |       18498 |   28498 |    1014 |       620 |      1000 | 33.880 (24)     | 0.518 (24) | 28.386 (24) |
|    8 |       11249 |       18670 |   28998 |    1014 |       620 |      1000 | 33.830 (17)     | 0.544 (17) | 28.260 (17) |
|    7 |       10211 |       18724 |   29000 |    1014 |       620 |      1000 | 32.580 (25)     | 0.512 (25) | 28.055 (35) |
|    5 |       10211 |       18724 |   29000 |    1014 |       620 |      1000 | 32.450 (29)     | 0.504 (29) | 28.257 (29) |
|    4 |       10211 |       18724 |   29000 |    1014 |       620 |      1000 | 32.010 (34)     | -          | 30.293 (32) |
|    2 |       10211 |       18724 |   29000 |    1014 |       620 |      1000 | 31.970 (25)     | 0.496 (25) | 30.425 (25) |
|   13 |       11249 |       19217 |   29000 |    1014 |       620 |      1000 | 30.920 (22)     | 0.495 (22) | 28.462 (28) |
|    3 |       10211 |       18724 |   29000 |    1014 |       620 |      1000 | 28.500 (8)      | 0.476 (8)  | 29.872 (5)  |
|    6 |       10211 |       18724 |   29000 |    1014 |       620 |      1000 | 28.460 (21)     | 0.493 (21) | 28.995 (24) |

|   id | file                                                                                                                                                                                                   |
|-----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   10 | /home/ozancag/wmt16/models/attention-wmt16-en-de-target-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050.log                                                 |
|   12 | /home/ozancag/wmt16/models/attention-wmt16-en-de/attention-embedding_dim_620-rnn_dim_1000-adadelta-decay_0.00050-gclip_-1.0-dropout_0.0-bs_32-valid_bleu-seed_1234-shuffle_no.log                      |
|    9 | /home/ozancag/wmt16/models/attention-wmt16-en-de-target-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050-fethi-filtered.log                                  |
|   11 | /home/ozancag/wmt16/models/attention-wmt16-en-de-target-lowercase/attention-embedding_dim_620-rnn_dim_1200-adadelta-bs_32-valid_bleu-decay_0.00050.log                                                 |
|    1 | /home/ozancag/wmt16/models/attention-liumtrain-en-de-target-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050.log                                             |
|    8 | /home/ozancag/wmt16/models/attention-wmt16-en-de-target-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050-all-new-tokenization.log                            |
|    7 | /home/ozancag/wmt16/models/attention-wmt16-en-de-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-decay_0.00050-gclip_5.0-dropout_0.0-bs_32-valid_bleu-seed_1234-shuffle_yes-inversion.log  |
|    5 | /home/ozancag/wmt16/models/attention-wmt16-en-de-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-decay_0.00050-gclip_-1.0-dropout_0.0-bs_32-valid_bleu-seed_1234-shuffle_yes-inversion.log |
|    4 | /home/ozancag/wmt16/models/attention-wmt16-en-de-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-decay_0.00050-gclip_-1.0-dropout_0.0-bs_32-valid_bleu-seed_1234-shuffle_no.log            |
|    2 | /home/ozancag/wmt16/models/attention-wmt16-en-de-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050-nosortshuffle.log                                          |
|   13 | /home/ozancag/wmt16/models/attention-wmt16-en-de/attention-embedding_dim_620-rnn_dim_1000-adadelta-decay_0.00050-gclip_5.0-dropout_0.0-bs_32-valid_bleu-seed_1234-shuffle_yes.log                      |
|    3 | /home/ozancag/wmt16/models/attention-wmt16-en-de-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-decay_0.00001-gclip_-1.0-dropout_0.0-bs_32-valid_bleu-seed_1234-shuffle_yes-inversion.log |
|    6 | /home/ozancag/wmt16/models/attention-wmt16-en-de-lowercase/attention-embedding_dim_620-rnn_dim_1000-adadelta-decay_0.00050-gclip_-1.0-dropout_0.0-bs_32-valid_bleu-seed_1234-shuffle_yes.log           |
