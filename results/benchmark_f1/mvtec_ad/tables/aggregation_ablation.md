# Image-score aggregation of the stored pixel anomaly maps (pixel_amaps, already smoothed with sigma 4)

| ('aggregation_rule', '')   |   ('image_auroc', 'mean') |   ('image_auroc', 'std') |   ('image_ap', 'mean') |   ('image_ap', 'std') |
|:---------------------------|--------------------------:|-------------------------:|-----------------------:|----------------------:|
| gaussian_pooled_max        |                  0.853741 |                 0.210779 |               0.927004 |              0.11053  |
| global_max                 |                  0.853516 |                 0.212952 |               0.925803 |              0.114604 |
| percentile_95              |                  0.7924   |                 0.22171  |               0.899916 |              0.115849 |
| percentile_99              |                  0.86346  |                 0.203553 |               0.933115 |              0.106522 |
| top_1_percent_mean         |                  0.868263 |                 0.205114 |               0.934888 |              0.108206 |