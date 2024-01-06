<!-- MarkdownTOC -->

- [n-1       @ extract_feature](#n_1___extract_featur_e_)
    - [1k-9600       @ n-1](#1k_9600___n_1_)
        - [interval-15       @ 1k-9600/n-1](#interval_15___1k_9600_n_1_)
        - [interval-6       @ 1k-9600/n-1](#interval_6___1k_9600_n_1_)
    - [25-2400       @ n-1](#25_2400___n_1_)
    - [25-2000-f       @ n-1](#25_2000_f___n_1_)
- [n-3       @ extract_feature](#n_3___extract_featur_e_)
    - [1k-9600       @ n-3](#1k_9600___n_3_)
        - [interval-15       @ 1k-9600/n-3](#interval_15___1k_9600_n_3_)
        - [interval-6       @ 1k-9600/n-3](#interval_6___1k_9600_n_3_)
    - [25-2400       @ n-3](#25_2400___n_3_)
    - [25-2000-f       @ n-3](#25_2000_f___n_3_)
- [n-5       @ extract_feature](#n_5___extract_featur_e_)
    - [25-2000-f       @ n-5](#25_2000_f___n_5_)

<!-- /MarkdownTOC -->

<a id="n_1___extract_featur_e_"></a>
# n-1       @ extract_feature-->dnc_gen
<a id="1k_9600___n_1_"></a>
## 1k-9600       @ n-1-->dnc_feat
<a id="interval_15___1k_9600_n_1_"></a>
### interval-15       @ 1k-9600/n-1-->dnc_feat
python3 run_mp.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var interval=15 n_proc=8 gpus=1,2
python3 run_mp.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var interval=15 n_proc=8 gpus=2
<a id="interval_6___1k_9600_n_1_"></a>
### interval-6       @ 1k-9600/n-1-->dnc_feat
python3 run_mp.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var interval=6 n_proc=8 gpus=1,2
python3 run_mp.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var interval=6 n_proc=8 gpus=1

<a id="25_2400___n_1_"></a>
## 25-2400       @ n-1-->dnc_feat
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2400_var interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2400_var interval=6
<a id="25_2000_f___n_1_"></a>
## 25-2000-f       @ n-1-->dnc_feat
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2000 interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2000 interval=6

<a id="n_3___extract_featur_e_"></a>
# n-3       @ extract_feature-->dnc_gen
<a id="1k_9600___n_3_"></a>
## 1k-9600       @ n-3-->dnc_feat
<a id="interval_15___1k_9600_n_3_"></a>
### interval-15       @ 1k-9600/n-3-->dnc_feat
python3 run_mp.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var interval=15 n_proc=8
<a id="interval_6___1k_9600_n_3_"></a>
### interval-6       @ 1k-9600/n-3-->dnc_feat
python3 run_mp.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var interval=6 n_proc=8
<a id="25_2400___n_3_"></a>
## 25-2400       @ n-3-->dnc_feat
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2400_var interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2400_var interval=6
<a id="25_2000_f___n_3_"></a>
## 25-2000-f       @ n-3-->dnc_feat
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2000 interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2000 interval=6

<a id="n_5___extract_featur_e_"></a>
# n-5       @ extract_feature-->dnc_gen
<a id="25_2000_f___n_5_"></a>
## 25-2000-f       @ n-5-->dnc_feat
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_5_25_2000 interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_5_25_2000 interval=6

