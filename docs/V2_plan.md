(jepa_robotics) tmainetucker@tmainetucker-MS-7E61:~/Repos/JEPA_Robotics$ python scripts/train.py --mode single --epochs 20 --fraction 0.1
Running in SINGLE mode for 20 epochs with 10% data. Using final V-JEPA backbone config.

Starting Training Run (Latent: 256, Epochs: 20, Patch: 16, MaskRatio: 0.75, SIGReg: 10.000)...
Loading REAL LeRobot SO100 Data OFFLINE from /home/tmainetucker/Repos/JEPA_Robotics/data/lerobot_so100... (Split: train, Fraction: 0.1)
Epoch 1/20: 0batch [00:00, ?batch/s]Loading REAL BridgeData V2 (Raw TFRecord) from /home/tmainetucker/Repos/JEPA_Robotics/data/bridge_data_v2... (Split: train, Fraction: 0.1)
Epoch 1/20: 671batch [03:14,  3.45batch/s, Avg L=0.006, Pos=0.015, Rot=0.279, Grp=0.176]

✅ Epoch 1 Train Completed - Avg Loss: 0.3263

Loading REAL BridgeData V2 (Raw TFRecord) from /home/tmainetucker/Repos/JEPA_Robotics/data/bridge_data_v2... (Split: val, Fraction: 0.1)
Loading REAL LeRobot SO100 Data OFFLINE from /home/tmainetucker/Repos/JEPA_Robotics/data/lerobot_so100... (Split: val, Fraction: 0.1)
Epoch 1 Validation: 88batch [00:47,  1.85batch/s, Val L=0.005, Pos=0.015, Rot=0.254, Grp=0.139]
\n🎯 Epoch 1 Validation Completed - Val Avg Loss: 0.0053 | Pos: 0.0146 | Rot: 0.2662 | Grp: 0.1579 | SMAC Score: 1.0342\n
Epoch 2/20: 671batch [02:12,  5.07batch/s, Avg L=0.002, Pos=0.010, Rot=0.257, Grp=0.147]

✅ Epoch 2 Train Completed - Avg Loss: 0.0028

Epoch 2 Validation: 88batch [00:28,  3.09batch/s, Val L=0.002, Pos=0.010, Rot=0.249, Grp=0.123]
\n🎯 Epoch 2 Validation Completed - Val Avg Loss: 0.0019 | Pos: 0.0103 | Rot: 0.2633 | Grp: 0.1323 | SMAC Score: 0.8925\n
Epoch 3/20: 671batch [02:14,  4.98batch/s, Avg L=0.009, Pos=0.009, Rot=0.244, Grp=0.100]

✅ Epoch 3 Train Completed - Avg Loss: 0.0040

Epoch 3 Validation: 88batch [00:28,  3.07batch/s, Val L=0.010, Pos=0.008, Rot=0.252, Grp=0.135]
\n🎯 Epoch 3 Validation Completed - Val Avg Loss: 0.0091 | Pos: 0.0091 | Rot: 0.2596 | Grp: 0.1264 | SMAC Score: 0.8499\n
Epoch 4/20: 671batch [02:12,  5.05batch/s, Avg L=0.184, Pos=0.010, Rot=0.244, Grp=0.116]

✅ Epoch 4 Train Completed - Avg Loss: 0.0812

Epoch 4 Validation: 88batch [00:29,  3.02batch/s, Val L=0.194, Pos=0.008, Rot=0.236, Grp=0.095]
\n🎯 Epoch 4 Validation Completed - Val Avg Loss: 0.2269 | Pos: 0.0088 | Rot: 0.2439 | Grp: 0.1158 | SMAC Score: 0.7979\n
Epoch 5/20: 671batch [02:11,  5.11batch/s, Avg L=1.131, Pos=0.009, Rot=0.233, Grp=0.099]

✅ Epoch 5 Train Completed - Avg Loss: 0.7928

Epoch 5 Validation: 88batch [00:28,  3.09batch/s, Val L=1.287, Pos=0.008, Rot=0.234, Grp=0.105]
\n🎯 Epoch 5 Validation Completed - Val Avg Loss: 1.1756 | Pos: 0.0092 | Rot: 0.2488 | Grp: 0.1276 | SMAC Score: 0.8364\n
Epoch 6/20: 671batch [02:11,  5.10batch/s, Avg L=0.397, Pos=0.010, Rot=0.222, Grp=0.092]

✅ Epoch 6 Train Completed - Avg Loss: 0.9320

Epoch 6 Validation: 88batch [00:28,  3.05batch/s, Val L=0.606, Pos=0.010, Rot=0.231, Grp=0.095]
\n🎯 Epoch 6 Validation Completed - Val Avg Loss: 0.5244 | Pos: 0.0098 | Rot: 0.2445 | Grp: 0.1123 | SMAC Score: 0.8109\n
Epoch 7/20: 671batch [02:12,  5.08batch/s, Avg L=1.486, Pos=0.008, Rot=0.273, Grp=0.164]

✅ Epoch 7 Train Completed - Avg Loss: 1.1883

Epoch 7 Validation: 88batch [00:28,  3.07batch/s, Val L=1.504, Pos=0.011, Rot=0.232, Grp=0.098]
\n🎯 Epoch 7 Validation Completed - Val Avg Loss: 1.3373 | Pos: 0.0127 | Rot: 0.2498 | Grp: 0.1125 | SMAC Score: 0.8789\n
Epoch 8/20: 671batch [02:11,  5.12batch/s, Avg L=1.515, Pos=0.014, Rot=0.237, Grp=0.110]

✅ Epoch 8 Train Completed - Avg Loss: 1.5379

Epoch 8 Validation: 88batch [00:28,  3.10batch/s, Val L=1.589, Pos=0.009, Rot=0.249, Grp=0.149]
\n🎯 Epoch 8 Validation Completed - Val Avg Loss: 1.4662 | Pos: 0.0105 | Rot: 0.2496 | Grp: 0.1220 | SMAC Score: 0.8532\n
Epoch 9/20: 671batch [02:08,  5.22batch/s, Avg L=1.668, Pos=0.012, Rot=0.231, Grp=0.101]

✅ Epoch 9 Train Completed - Avg Loss: 1.5360

Epoch 9 Validation: 88batch [00:28,  3.09batch/s, Val L=2.077, Pos=0.014, Rot=0.288, Grp=0.238]
\n🎯 Epoch 9 Validation Completed - Val Avg Loss: 1.8903 | Pos: 0.0159 | Rot: 0.2984 | Grp: 0.2307 | SMAC Score: 1.2570\n
Epoch 10/20: 671batch [02:13,  5.04batch/s, Avg L=1.491, Pos=0.008, Rot=0.237, Grp=0.110]

✅ Epoch 10 Train Completed - Avg Loss: 1.7058

Epoch 10 Validation: 88batch [00:29,  3.02batch/s, Val L=2.279, Pos=0.015, Rot=0.218, Grp=0.068]
\n🎯 Epoch 10 Validation Completed - Val Avg Loss: 2.2390 | Pos: 0.0154 | Rot: 0.2372 | Grp: 0.1019 | SMAC Score: 0.8917\n
Epoch 11/20: 671batch [02:08,  5.23batch/s, Avg L=3.075, Pos=0.009, Rot=0.222, Grp=0.085]

✅ Epoch 11 Train Completed - Avg Loss: 1.9334

Epoch 11 Validation: 88batch [00:28,  3.09batch/s, Val L=1.949, Pos=0.007, Rot=0.222, Grp=0.073]
\n🎯 Epoch 11 Validation Completed - Val Avg Loss: 1.9525 | Pos: 0.0086 | Rot: 0.2387 | Grp: 0.1046 | SMAC Score: 0.7634\n
Epoch 12/20: 671batch [02:09,  5.17batch/s, Avg L=1.797, Pos=0.007, Rot=0.209, Grp=0.079]

✅ Epoch 12 Train Completed - Avg Loss: 2.1359

Epoch 12 Validation: 88batch [00:30,  2.88batch/s, Val L=2.246, Pos=0.010, Rot=0.199, Grp=0.050]
\n🎯 Epoch 12 Validation Completed - Val Avg Loss: 2.1370 | Pos: 0.0166 | Rot: 0.2339 | Grp: 0.0955 | SMAC Score: 0.8982\n
Epoch 13/20: 671batch [02:10,  5.15batch/s, Avg L=1.634, Pos=0.013, Rot=0.203, Grp=0.069]

✅ Epoch 13 Train Completed - Avg Loss: 1.7353

Epoch 13 Validation: 88batch [00:29,  3.03batch/s, Val L=1.906, Pos=0.017, Rot=0.245, Grp=0.068]
\n🎯 Epoch 13 Validation Completed - Val Avg Loss: 1.7324 | Pos: 0.0177 | Rot: 0.2469 | Grp: 0.0952 | SMAC Score: 0.9389\n
Epoch 14/20: 671batch [02:08,  5.23batch/s, Avg L=1.869, Pos=0.007, Rot=0.209, Grp=0.065]

✅ Epoch 14 Train Completed - Avg Loss: 1.8850

Epoch 14 Validation: 88batch [00:28,  3.06batch/s, Val L=2.119, Pos=0.012, Rot=0.202, Grp=0.037]
\n🎯 Epoch 14 Validation Completed - Val Avg Loss: 2.0584 | Pos: 0.0105 | Rot: 0.2203 | Grp: 0.0717 | SMAC Score: 0.7055\n
Epoch 15/20: 671batch [02:11,  5.11batch/s, Avg L=1.518, Pos=0.007, Rot=0.196, Grp=0.075]

✅ Epoch 15 Train Completed - Avg Loss: 1.9886

Epoch 15 Validation: 88batch [00:29,  2.99batch/s, Val L=3.173, Pos=0.012, Rot=0.222, Grp=0.092]
\n🎯 Epoch 15 Validation Completed - Val Avg Loss: 2.7787 | Pos: 0.0115 | Rot: 0.2315 | Grp: 0.0982 | SMAC Score: 0.7972\n
Epoch 16/20: 671batch [02:17,  4.88batch/s, Avg L=1.934, Pos=0.006, Rot=0.204, Grp=0.072]

✅ Epoch 16 Train Completed - Avg Loss: 2.0452

Epoch 16 Validation: 88batch [00:28,  3.09batch/s, Val L=2.787, Pos=0.011, Rot=0.211, Grp=0.050]
\n🎯 Epoch 16 Validation Completed - Val Avg Loss: 2.3495 | Pos: 0.0113 | Rot: 0.2338 | Grp: 0.0825 | SMAC Score: 0.7645\n
Epoch 17/20: 671batch [02:08,  5.24batch/s, Avg L=1.552, Pos=0.016, Rot=0.220, Grp=0.174]

✅ Epoch 17 Train Completed - Avg Loss: 2.0952

Epoch 17 Validation: 88batch [00:28,  3.05batch/s, Val L=2.875, Pos=0.009, Rot=0.215, Grp=0.049]
\n🎯 Epoch 17 Validation Completed - Val Avg Loss: 2.3795 | Pos: 0.0098 | Rot: 0.2418 | Grp: 0.0917 | SMAC Score: 0.7657\n
Epoch 18/20: 671batch [02:09,  5.17batch/s, Avg L=1.457, Pos=0.012, Rot=0.218, Grp=0.091]

✅ Epoch 18 Train Completed - Avg Loss: 2.0800

Epoch 18 Validation: 88batch [00:29,  2.97batch/s, Val L=3.484, Pos=0.010, Rot=0.201, Grp=0.039]
\n🎯 Epoch 18 Validation Completed - Val Avg Loss: 3.0495 | Pos: 0.0098 | Rot: 0.2328 | Grp: 0.0817 | SMAC Score: 0.7320\n
Epoch 19/20: 671batch [02:08,  5.21batch/s, Avg L=1.983, Pos=0.012, Rot=0.210, Grp=0.083]

✅ Epoch 19 Train Completed - Avg Loss: 2.1332

Epoch 19 Validation: 88batch [00:28,  3.08batch/s, Val L=2.576, Pos=0.009, Rot=0.239, Grp=0.067]
\n🎯 Epoch 19 Validation Completed - Val Avg Loss: 2.1499 | Pos: 0.0101 | Rot: 0.2609 | Grp: 0.1093 | SMAC Score: 0.8374\n
Epoch 20/20: 671batch [02:14,  4.99batch/s, Avg L=1.616, Pos=0.007, Rot=0.190, Grp=0.080]

✅ Epoch 20 Train Completed - Avg Loss: 2.1723

Epoch 20 Validation: 88batch [00:29,  3.03batch/s, Val L=2.892, Pos=0.009, Rot=0.214, Grp=0.055]
\n🎯 Epoch 20 Validation Completed - Val Avg Loss: 2.3622 | Pos: 0.0097 | Rot: 0.2345 | Grp: 0.0863 | SMAC Score: 0.7428\n

[INFO] Saving final model checkpoint to checkpoints/v1_jepa_backbone...
[INFO] Model successfully saved to checkpoints/v1_jepa_backbone/v1_weights.msgpack!

\nSingle run completed. Final Loss: 0.7428
terminate called without an active exception
Aborted (core dumped)