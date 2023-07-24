<!-- MarkdownTOC -->

- [n-1](#n_1_)
  - [25-2400       @ n-1](#25_2400___n_1_)
    - [all_seq__6_6       @ 25-2400/n-1](#all_seq_6_6___25_2400_n_1_)
    - [all_seq__6_15       @ 25-2400/n-1](#all_seq_6_15___25_2400_n_1_)
    - [seq_0_25__6_15       @ 25-2400/n-1](#seq_0_25_6_15___25_2400_n_1_)
- [n-3](#n_3_)
  - [1k-9600       @ n-3](#1k_9600___n_3_)
    - [all_seq__6_15       @ 1k-9600/n-3](#all_seq_6_15___1k_9600_n_3_)
    - [all_seq__6_6       @ 1k-9600/n-3](#all_seq_6_6___1k_9600_n_3_)
  - [25-2400       @ n-3](#25_2400___n_3_)
    - [all_seq__6_15       @ 25-2400/n-3](#all_seq_6_15___25_2400_n_3_)
    - [seq_0_25__6_15       @ 25-2400/n-3](#seq_0_25_6_15___25_2400_n_3_)
- [fixed](#fixed_)
  - [n-3       @ fixed](#n_3___fixe_d_)
    - [all_seq__6_15       @ n-3/fixed](#all_seq_6_15___n_3_fixe_d_)
    - [seq_0_25__6_15       @ n-3/fixed](#seq_0_25_6_15___n_3_fixe_d_)
  - [n-5       @ fixed](#n_5___fixe_d_)
    - [seq_0__6_15       @ n-5/fixed](#seq_0_6_15___n_5_fixe_d_)
    - [seq_0_25__6_15       @ n-5/fixed](#seq_0_25_6_15___n_5_fixe_d_)
    - [seq_0_25__6_6       @ n-5/fixed](#seq_0_25_6_6___n_5_fixe_d_)

<!-- /MarkdownTOC -->
<a id="n_1_"></a>
# n-1
<a id="25_2400___n_1_"></a>
## 25-2400       @ n-1-->dnc
<a id="all_seq_6_6___25_2400_n_1_"></a>
### all_seq__6_6       @ 25-2400/n-1-->dnc
CUDA_VISIBLE_DEVICES=0,1 python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_25_2400_var/all_seq__6_6.cfg --batch_size 32 --cuda

python3 test.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_25_2400_var/all_seq__6_15.cfg --start_from log//model_epoch_$epoch.t7 --id $id-$epoch --cuda

<a id="all_seq_6_15___25_2400_n_1_"></a>
### all_seq__6_15       @ 25-2400/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_25_2400_var/all_seq__6_15.cfg --batch_size 32 --cuda

<a id="seq_0_25_6_15___25_2400_n_1_"></a>
### seq_0_25__6_15       @ 25-2400/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_25_2400_var/seq_0_25__6_15.cfg --checkpoint_path log/MNIST_MOT_RGB_512x512_1_25_2400_var/seq_0_25__6_15 --batch_size 32 --cuda

<a id="n_3_"></a>
# n-3
<a id="1k_9600___n_3_"></a>
## 1k-9600       @ n-3-->dnc
<a id="all_seq_6_15___1k_9600_n_3_"></a>
### all_seq__6_15       @ 1k-9600/n-3-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq__6_15.cfg --batch_size 32 --num_workers 4
<a id="all_seq_6_6___1k_9600_n_3_"></a>
### all_seq__6_6       @ 1k-9600/n-3-->dnc
CUDA_VISIBLE_DEVICES=0 python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq__6_6.cfg --batch_size 28 --num_workers 4 --n_proc 12

```
ulimit -n 40960
```

<a id="25_2400___n_3_"></a>
## 25-2400       @ n-3-->dnc
<a id="all_seq_6_15___25_2400_n_3_"></a>
### all_seq__6_15       @ 25-2400/n-3-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_25_2400_var/all_seq__6_15.cfg --batch_size 64 --cuda
<a id="seq_0_25_6_15___25_2400_n_3_"></a>
### seq_0_25__6_15       @ 25-2400/n-3-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_25_2400_var/seq_0_25__6_15.cfg --batch_size 32 --cuda


<a id="fixed_"></a>
# fixed
<a id="n_3___fixe_d_"></a>
## n-3       @ fixed-->dnc
<a id="all_seq_6_15___n_3_fixe_d_"></a>
### all_seq__6_15       @ n-3/fixed-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_25_2000/all_seq__6_15.cfg --checkpoint_path log/MNIST_MOT_RGB_512x512_3_25_2000/all_seq__6_15 --batch_size 4 --cuda
<a id="seq_0_25_6_15___n_3_fixe_d_"></a>
### seq_0_25__6_15       @ n-3/fixed-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_25_2000/seq_0_25__6_15.cfg --checkpoint_path log/MNIST_MOT_RGB_512x512_3_25_2000/seq_0_25__6_15 --batch_size 16 --cuda

<a id="n_5___fixe_d_"></a>
## n-5       @ fixed-->dnc
<a id="seq_0_6_15___n_5_fixe_d_"></a>
### seq_0__6_15       @ n-5/fixed-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_5_25_2000/seq_0__6_15.cfg --checkpoint_path log/MNIST_MOT_RGB_512x512_5_25_2000/seq_0__6_15 --batch_size 4 --cuda
<a id="seq_0_25_6_15___n_5_fixe_d_"></a>
### seq_0_25__6_15       @ n-5/fixed-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_5_25_2000/seq_0_25__6_15.cfg --checkpoint_path log/MNIST_MOT_RGB_512x512_5_25_2000/seq_0_25__6_15 --batch_size 4 --cuda
<a id="seq_0_25_6_6___n_5_fixe_d_"></a>
### seq_0_25__6_6       @ n-5/fixed-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_5_25_2000/seq_0_25__6_6.cfg --checkpoint_path log/MNIST_MOT_RGB_512x512_5_25_2000/seq_0_25__6_6 --batch_size 4 
























































































