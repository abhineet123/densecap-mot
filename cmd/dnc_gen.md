<!-- MarkdownTOC -->

- [n-1](#n_1_)
    - [25-2000-f       @ n-1](#25_2000_f___n_1_)
        - [all       @ 25-2000-f/n-1](#all___25_2000_f_n_1_)
        - [seq_0_25       @ 25-2000-f/n-1](#seq_0_25___25_2000_f_n_1_)
    - [25-2400       @ n-1](#25_2400___n_1_)
        - [all       @ 25-2400/n-1](#all___25_2400_n_1_)
        - [seq_0_25       @ 25-2400/n-1](#seq_0_25___25_2400_n_1_)
    - [1k-9600       @ n-1](#1k_9600___n_1_)
        - [1k-9600-dummy-100_2       @ 1k-9600/n-1](#1k_9600_dummy_100_2___1k_9600_n_1_)
    - [100-960       @ n-1](#100_960___n_1_)
        - [slide-24       @ 100-960/n-1](#slide_24___100_960_n_1_)
            - [64x64       @ slide-24/100-960/n-1](#64x64___slide_24_100_960_n_1_)
        - [slide-16       @ 100-960/n-1](#slide_16___100_960_n_1_)
            - [64x64       @ slide-16/100-960/n-1](#64x64___slide_16_100_960_n_1_)
        - [slide-12       @ 100-960/n-1](#slide_12___100_960_n_1_)
            - [64x64       @ slide-12/100-960/n-1](#64x64___slide_12_100_960_n_1_)
- [n-3](#n_3_)
    - [25-2000-f       @ n-3](#25_2000_f___n_3_)
        - [all       @ 25-2000-f/n-3](#all___25_2000_f_n_3_)
        - [seq_0_25       @ 25-2000-f/n-3](#seq_0_25___25_2000_f_n_3_)
    - [25-2400       @ n-3](#25_2400___n_3_)
        - [all       @ 25-2400/n-3](#all___25_2400_n_3_)
        - [seq_0_25       @ 25-2400/n-3](#seq_0_25___25_2400_n_3_)
    - [1k-9600       @ n-3](#1k_9600___n_3_)
        - [slide-15-480       @ 1k-9600/n-3](#slide_15_480___1k_9600_n_3_)
            - [diff       @ slide-15-480/1k-9600/n-3](#diff___slide_15_480_1k_9600_n_3_)
            - [diff-sample       @ slide-15-480/1k-9600/n-3](#diff_sample___slide_15_480_1k_9600_n_3_)
        - [slide-6-480       @ 1k-9600/n-3](#slide_6_480___1k_9600_n_3_)
            - [diff-sample       @ slide-6-480/1k-9600/n-3](#diff_sample___slide_6_480_1k_9600_n_3_)
- [n-5](#n_5_)
    - [25-2000-f       @ n-5](#25_2000_f___n_5_)
        - [all       @ 25-2000-f/n-5](#all___25_2000_f_n_5_)
        - [seq_49       @ 25-2000-f/n-5](#seq_49___25_2000_f_n_5_)
        - [seq_0       @ 25-2000-f/n-5](#seq_0___25_2000_f_n_5_)
        - [seq_0_25       @ 25-2000-f/n-5](#seq_0_25___25_2000_f_n_5_)
- [DETRAC       @ mot_to_dnc](#detrac___mot_to_dnc_)

<!-- /MarkdownTOC -->

<a id="n_1_"></a>
# n-1       
<a id="25_2000_f___n_1_"></a>
## 25-2000-f       @ n-1-->dnc_gen
<a id="all___25_2000_f_n_1_"></a>
### all       @ 25-2000-f/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2000
<a id="seq_0_25___25_2000_f_n_1_"></a>
### seq_0_25       @ 25-2000-f/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2000 seq=0,25

<a id="25_2400___n_1_"></a>
## 25-2400       @ n-1-->dnc_gen
<a id="all___25_2400_n_1_"></a>
### all       @ 25-2400/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2400_var
<a id="seq_0_25___25_2400_n_1_"></a>
### seq_0_25       @ 25-2400/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2400_var seq=0,25

<a id="1k_9600___n_1_"></a>
## 1k-9600       @ n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var n_proc=12
<a id="1k_9600_dummy_100_2___1k_9600_n_1_"></a>
### 1k-9600-dummy-100_2       @ 1k-9600/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var start_seq=0,1000 end_seq=99,1099 @slide size=480 num=2 sample=1 @ fixed_traj_len=20

python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var start_seq=1000 end_seq=1002 @slide size=480 num=2 sample=1 @ fixed_traj_len=20

<a id="100_960___n_1_"></a>
## 100-960       @ n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var @slide size=480 sample=1 @ fixed_traj_len=20

python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var @slide size=480 sample=1 @ fixed_traj_len=20 start_seq=0,100 end_seq=1,101

__vocab_fmt__
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var sample=1 start_seq=100 end_seq=100 vocab_fmt=1 @slide size=480

python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var sample=1 start_seq=100 end_seq=100 vocab_fmt=2 @slide size=480

<a id="slide_24___100_960_n_1_"></a>
### slide-24       @ 100-960/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var @slide size=24 sample=1
__dbg__
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var @slide size=24 sample=1 @ start_seq=0,100 end_seq=1,101

<a id="64x64___slide_24_100_960_n_1_"></a>
#### 64x64       @ slide-24/100-960/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var grid_res=64,64 no_repeat=1 @slide size=24 sample=1 @ min_traj_len=6 n_proc=12
__dbg__
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var grid_res=64,64 no_repeat=1 @slide size=24 sample=1 @ min_traj_len=6 n_proc=12 start_seq=0,100 end_seq=5,105

<a id="slide_16___100_960_n_1_"></a>
### slide-16       @ 100-960/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var @slide size=16 sample=1 @ n_proc=12
<a id="64x64___slide_16_100_960_n_1_"></a>
#### 64x64       @ slide-16/100-960/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var grid_res=64,64 no_repeat=1 @slide size=16 sample=1 @ min_traj_len=6 n_proc=12
__dbg__
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var grid_res=64,64 no_repeat=1 @slide size=16 sample=1 @ min_traj_len=6 n_proc=12 start_seq=0,100 end_seq=5,105

<a id="slide_12___100_960_n_1_"></a>
### slide-12       @ 100-960/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var @slide size=12 sample=1
__dbg__
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var @slide size=12 sample=1 @ start_seq=0,100 end_seq=1,101

<a id="64x64___slide_12_100_960_n_1_"></a>
#### 64x64       @ slide-12/100-960/n-1-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var grid_res=64,64 no_repeat=1 @slide size=12 sample=1 @ min_traj_len=6 n_proc=12
__dbg__
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_100_960_var grid_res=64,64 no_repeat=1 @slide size=120 sample=1 @ start_seq=0 end_seq=0 vis=0 min_traj_len=60

python3 dnc_to_mot.py json=/data/MNIST_MOT_RGB_512x512_1_100_960_var/seq_0_1_slide_120_64x64_no_repeat.json set=MNIST_MOT_RGB_512x512_1_100_960_var grid_res=64,64 seq=100 vis=1 @slide size=120 sample=1


<a id="n_3_"></a>
# n-3      
<a id="25_2000_f___n_3_"></a>
## 25-2000-f       @ n-3-->dnc_gen
<a id="all___25_2000_f_n_3_"></a>
### all       @ 25-2000-f/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2000
<a id="seq_0_25___25_2000_f_n_3_"></a>
### seq_0_25       @ 25-2000-f/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2000 seq=0,25

<a id="25_2400___n_3_"></a>
## 25-2400       @ n-3-->dnc_gen
<a id="all___25_2400_n_3_"></a>
### all       @ 25-2400/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2400_var
<a id="seq_0_25___25_2400_n_3_"></a>
### seq_0_25       @ 25-2400/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2400_var seq=0,25

<a id="1k_9600___n_3_"></a>
## 1k-9600       @ n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var n_proc=12
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var seq=782

<a id="slide_15_480___1k_9600_n_3_"></a>
### slide-15-480       @ 1k-9600/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var start_seq=0,1000 end_seq=9,1009 @slide size=480 num=2 sample=1

<a id="diff___slide_15_480_1k_9600_n_3_"></a>
#### diff       @ slide-15-480/1k-9600/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var @slide sample=15 size=480 @ n_proc=12 vocab_fmt=1 max_diff=99 n_proc=12 sample_traj=0 seq=1000
<a id="diff_sample___slide_15_480_1k_9600_n_3_"></a>
#### diff-sample       @ slide-15-480/1k-9600/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var @slide sample=15 size=480 @ n_proc=12 vocab_fmt=1 max_diff=99 n_proc=12 sample_traj=1 seq=999

python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var @slide sample=15 size=480 @ n_proc=12 vocab_fmt=1 max_diff=99 n_proc=12 sample_traj=1 vis=1 seq=782

<a id="slide_6_480___1k_9600_n_3_"></a>
### slide-6-480       @ 1k-9600/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var @slide sample=6 size=480 @ n_proc=12
<a id="diff_sample___slide_6_480_1k_9600_n_3_"></a>
#### diff-sample       @ slide-6-480/1k-9600/n-3-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var @slide sample=6 size=480 @ n_proc=12 vocab_fmt=1 max_diff=99 n_proc=12 sample_traj=1

<a id="n_5_"></a>
# n-5
<a id="25_2000_f___n_5_"></a>
## 25-2000-f       @ n-5-->dnc_gen
<a id="all___25_2000_f_n_5_"></a>
### all       @ 25-2000-f/n-5-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000
<a id="seq_49___25_2000_f_n_5_"></a>
### seq_49       @ 25-2000-f/n-5-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=49 vis=1
<a id="seq_0___25_2000_f_n_5_"></a>
### seq_0       @ 25-2000-f/n-5-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=0
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=0 frame_gap=20 win_size=100
<a id="seq_0_25___25_2000_f_n_5_"></a>
### seq_0_25       @ 25-2000-f/n-5-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=0,25

<a id="detrac___mot_to_dnc_"></a>
# DETRAC       @ mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py seq_set=7 seq=0


