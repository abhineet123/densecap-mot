<!-- MarkdownTOC -->

- [gen_mnist_mot](#gen_mnist_mot_)
    - [128       @ gen_mnist_mot](#128___gen_mnist_mo_t_)
        - [25-2000-f       @ 128/gen_mnist_mot](#25_2000_f___128_gen_mnist_mo_t_)
            - [debug       @ 25-2000-f/128/gen_mnist_mot](#debug___25_2000_f_128_gen_mnist_mo_t_)
    - [512       @ gen_mnist_mot](#512___gen_mnist_mo_t_)
        - [n-1       @ 512/gen_mnist_mot](#n_1___512_gen_mnist_mo_t_)
            - [25-2000-f       @ n-1/512/gen_mnist_mot](#25_2000_f___n_1_512_gen_mnist_mo_t_)
            - [25-2400       @ n-1/512/gen_mnist_mot](#25_2400___n_1_512_gen_mnist_mo_t_)
            - [1k-9600       @ n-1/512/gen_mnist_mot](#1k_9600___n_1_512_gen_mnist_mo_t_)
        - [n-2       @ 512/gen_mnist_mot](#n_2___512_gen_mnist_mo_t_)
            - [25-2000-f       @ n-2/512/gen_mnist_mot](#25_2000_f___n_2_512_gen_mnist_mo_t_)
        - [n-3       @ 512/gen_mnist_mot](#n_3___512_gen_mnist_mo_t_)
            - [25-2000-f       @ n-3/512/gen_mnist_mot](#25_2000_f___n_3_512_gen_mnist_mo_t_)
            - [25-2400       @ n-3/512/gen_mnist_mot](#25_2400___n_3_512_gen_mnist_mo_t_)
            - [1k-9600       @ n-3/512/gen_mnist_mot](#1k_9600___n_3_512_gen_mnist_mo_t_)
        - [n-5       @ 512/gen_mnist_mot](#n_5___512_gen_mnist_mo_t_)
            - [25-2000-f       @ n-5/512/gen_mnist_mot](#25_2000_f___n_5_512_gen_mnist_mo_t_)
    - [minst_dd_rgb       @ gen_mnist_mot](#minst_dd_rgb___gen_mnist_mo_t_)
- [extract_feature](#extract_feature_)
    - [n-1       @ extract_feature](#n_1___extract_featur_e_)
        - [1k-9600       @ n-1/extract_feature](#1k_9600___n_1_extract_featur_e_)
            - [interval-15       @ 1k-9600/n-1/extract_feature](#interval_15___1k_9600_n_1_extract_featur_e_)
            - [interval-6       @ 1k-9600/n-1/extract_feature](#interval_6___1k_9600_n_1_extract_featur_e_)
        - [25-2400       @ n-1/extract_feature](#25_2400___n_1_extract_featur_e_)
        - [25-2000-f       @ n-1/extract_feature](#25_2000_f___n_1_extract_featur_e_)
    - [n-3       @ extract_feature](#n_3___extract_featur_e_)
        - [1k-9600       @ n-3/extract_feature](#1k_9600___n_3_extract_featur_e_)
            - [interval-15       @ 1k-9600/n-3/extract_feature](#interval_15___1k_9600_n_3_extract_featur_e_)
            - [interval-6       @ 1k-9600/n-3/extract_feature](#interval_6___1k_9600_n_3_extract_featur_e_)
        - [25-2400       @ n-3/extract_feature](#25_2400___n_3_extract_featur_e_)
        - [25-2000-f       @ n-3/extract_feature](#25_2000_f___n_3_extract_featur_e_)
    - [n-5       @ extract_feature](#n_5___extract_featur_e_)
        - [25-2000-f       @ n-5/extract_feature](#25_2000_f___n_5_extract_featur_e_)
- [mot_to_dnc](#mot_to_dn_c_)
    - [n-1       @ mot_to_dnc](#n_1___mot_to_dnc_)
        - [25-2000-f       @ n-1/mot_to_dnc](#25_2000_f___n_1_mot_to_dnc_)
            - [all       @ 25-2000-f/n-1/mot_to_dnc](#all___25_2000_f_n_1_mot_to_dnc_)
            - [seq_0_25       @ 25-2000-f/n-1/mot_to_dnc](#seq_0_25___25_2000_f_n_1_mot_to_dnc_)
        - [25-2400       @ n-1/mot_to_dnc](#25_2400___n_1_mot_to_dnc_)
            - [all       @ 25-2400/n-1/mot_to_dnc](#all___25_2400_n_1_mot_to_dnc_)
            - [seq_0_25       @ 25-2400/n-1/mot_to_dnc](#seq_0_25___25_2400_n_1_mot_to_dnc_)
        - [1k-9600       @ n-1/mot_to_dnc](#1k_9600___n_1_mot_to_dnc_)
    - [n-2       @ mot_to_dnc](#n_2___mot_to_dnc_)
        - [25-2000-f       @ n-2/mot_to_dnc](#25_2000_f___n_2_mot_to_dnc_)
            - [all       @ 25-2000-f/n-2/mot_to_dnc](#all___25_2000_f_n_2_mot_to_dnc_)
            - [seq_0_25       @ 25-2000-f/n-2/mot_to_dnc](#seq_0_25___25_2000_f_n_2_mot_to_dnc_)
    - [n-3       @ mot_to_dnc](#n_3___mot_to_dnc_)
        - [25-2000-f       @ n-3/mot_to_dnc](#25_2000_f___n_3_mot_to_dnc_)
            - [all       @ 25-2000-f/n-3/mot_to_dnc](#all___25_2000_f_n_3_mot_to_dnc_)
            - [seq_0_25       @ 25-2000-f/n-3/mot_to_dnc](#seq_0_25___25_2000_f_n_3_mot_to_dnc_)
        - [25-2400       @ n-3/mot_to_dnc](#25_2400___n_3_mot_to_dnc_)
            - [all       @ 25-2400/n-3/mot_to_dnc](#all___25_2400_n_3_mot_to_dnc_)
            - [seq_0_25       @ 25-2400/n-3/mot_to_dnc](#seq_0_25___25_2400_n_3_mot_to_dnc_)
        - [1k-9600       @ n-3/mot_to_dnc](#1k_9600___n_3_mot_to_dnc_)
    - [n-5       @ mot_to_dnc](#n_5___mot_to_dnc_)
        - [25-2000-f       @ n-5/mot_to_dnc](#25_2000_f___n_5_mot_to_dnc_)
            - [all       @ 25-2000-f/n-5/mot_to_dnc](#all___25_2000_f_n_5_mot_to_dnc_)
            - [seq_49       @ 25-2000-f/n-5/mot_to_dnc](#seq_49___25_2000_f_n_5_mot_to_dnc_)
            - [seq_0       @ 25-2000-f/n-5/mot_to_dnc](#seq_0___25_2000_f_n_5_mot_to_dnc_)
            - [seq_0_25       @ 25-2000-f/n-5/mot_to_dnc](#seq_0_25___25_2000_f_n_5_mot_to_dnc_)
    - [DETRAC       @ mot_to_dnc](#detrac___mot_to_dnc_)

<!-- /MarkdownTOC -->

<a id="gen_mnist_mot_"></a>
# gen_mnist_mot

<a id="128___gen_mnist_mo_t_"></a>
## 128       @ gen_mnist_mot-->dnc_gen
<a id="25_2000_f___128_gen_mnist_mo_t_"></a>
### 25-2000-f       @ 128/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=25 n_frames=2e3
<a id="debug___25_2000_f_128_gen_mnist_mo_t_"></a>
#### debug       @ 25-2000-f/128/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=3 n_frames=20

<a id="512___gen_mnist_mo_t_"></a>
## 512       @ gen_mnist_mot-->dnc_gen
<a id="n_1___512_gen_mnist_mo_t_"></a>
### n-1       @ 512/gen_mnist_mot-->dnc_gen
<a id="25_2000_f___n_1_512_gen_mnist_mo_t_"></a>
#### 25-2000-f       @ n-1/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=25 n_frames=2e3 img_h=512 n_objs=1 velocity=10.6
<a id="25_2400___n_1_512_gen_mnist_mo_t_"></a>
#### 25-2400       @ n-1/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=25 n_frames=2400 img_h=512 n_objs=1
<a id="1k_9600___n_1_512_gen_mnist_mo_t_"></a>
#### 1k-9600       @ n-1/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=1000 n_frames=9600 img_h=512 n_objs=1

<a id="n_2___512_gen_mnist_mo_t_"></a>
### n-2       @ 512/gen_mnist_mot-->dnc_gen
<a id="25_2000_f___n_2_512_gen_mnist_mo_t_"></a>
#### 25-2000-f       @ n-2/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=25 n_frames=2e3 img_h=512 n_objs=2 velocity=10.6

<a id="n_3___512_gen_mnist_mo_t_"></a>
### n-3       @ 512/gen_mnist_mot-->dnc_gen
<a id="25_2000_f___n_3_512_gen_mnist_mo_t_"></a>
#### 25-2000-f       @ n-3/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=25 n_frames=2e3 img_h=512 n_objs=3 velocity=10.6
<a id="25_2400___n_3_512_gen_mnist_mo_t_"></a>
#### 25-2400       @ n-3/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=25 n_frames=2400 img_h=512 n_objs=3
<a id="1k_9600___n_3_512_gen_mnist_mo_t_"></a>
#### 1k-9600       @ n-3/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=1000 n_frames=9600 img_h=512 n_objs=3

<a id="n_5___512_gen_mnist_mo_t_"></a>
### n-5       @ 512/gen_mnist_mot-->dnc_gen
<a id="25_2000_f___n_5_512_gen_mnist_mo_t_"></a>
#### 25-2000-f       @ n-5/512/gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=0 n_seq=25 n_frames=2e3 img_h=512 n_objs=5 velocity=10.6

<a id="minst_dd_rgb___gen_mnist_mo_t_"></a>
## minst_dd_rgb       @ gen_mnist_mot-->dnc_gen
python3 gen_mnist_mot.py show_img=1 n_train_seq=10 n_test_seq=10 n_frames=1 img_h=256 n_objs=2 max_obj_size=128 min_obj_size=32 batch_size=1 appear_interval=0 birth_prob=1

python3 gen_mnist_mot.py show_img=0 n_seq=1000 n_frames=1000 img_h=256 n_objs=2 velocity=10.6 max_obj_size=128 min_obj_size=32

<a id="extract_feature_"></a>
# extract_feature

<a id="n_1___extract_featur_e_"></a>
## n-1       @ extract_feature-->dnc_gen
<a id="1k_9600___n_1_extract_featur_e_"></a>
### 1k-9600       @ n-1/extract_feature-->dnc_gen
<a id="interval_15___1k_9600_n_1_extract_featur_e_"></a>
#### interval-15       @ 1k-9600/n-1/extract_feature-->dnc_gen
python3 run_mp.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var interval=15 n_proc=8
<a id="interval_6___1k_9600_n_1_extract_featur_e_"></a>
#### interval-6       @ 1k-9600/n-1/extract_feature-->dnc_gen
python3 run_mp.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var interval=6 n_proc=8

<a id="25_2400___n_1_extract_featur_e_"></a>
### 25-2400       @ n-1/extract_feature-->dnc_gen
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2400_var interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2400_var interval=6
<a id="25_2000_f___n_1_extract_featur_e_"></a>
### 25-2000-f       @ n-1/extract_feature-->dnc_gen
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2000 interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_1_25_2000 interval=6

<a id="n_3___extract_featur_e_"></a>
## n-3       @ extract_feature-->dnc_gen
<a id="1k_9600___n_3_extract_featur_e_"></a>
### 1k-9600       @ n-3/extract_feature-->dnc_gen
<a id="interval_15___1k_9600_n_3_extract_featur_e_"></a>
#### interval-15       @ 1k-9600/n-3/extract_feature-->dnc_gen
python3 run_mp.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var interval=15 n_proc=8
<a id="interval_6___1k_9600_n_3_extract_featur_e_"></a>
#### interval-6       @ 1k-9600/n-3/extract_feature-->dnc_gen
python3 run_mp.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var interval=6 n_proc=8
<a id="25_2400___n_3_extract_featur_e_"></a>
### 25-2400       @ n-3/extract_feature-->dnc_gen
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2400_var interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2400_var interval=6
<a id="25_2000_f___n_3_extract_featur_e_"></a>
### 25-2000-f       @ n-3/extract_feature-->dnc_gen
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2000 interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_3_25_2000 interval=6

<a id="n_5___extract_featur_e_"></a>
## n-5       @ extract_feature-->dnc_gen
<a id="25_2000_f___n_5_extract_featur_e_"></a>
### 25-2000-f       @ n-5/extract_feature-->dnc_gen
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_5_25_2000 interval=15
python3 extract_feature.py set=MNIST_MOT_RGB_512x512_5_25_2000 interval=6

<a id="mot_to_dn_c_"></a>
# mot_to_dnc
<a id="n_1___mot_to_dnc_"></a>
## n-1       @ mot_to_dnc-->dnc_gen

<a id="25_2000_f___n_1_mot_to_dnc_"></a>
### 25-2000-f       @ n-1/mot_to_dnc-->dnc_gen
<a id="all___25_2000_f_n_1_mot_to_dnc_"></a>
#### all       @ 25-2000-f/n-1/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2000
<a id="seq_0_25___25_2000_f_n_1_mot_to_dnc_"></a>
#### seq_0_25       @ 25-2000-f/n-1/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2000 seq=0,25

<a id="25_2400___n_1_mot_to_dnc_"></a>
### 25-2400       @ n-1/mot_to_dnc-->dnc_gen
<a id="all___25_2400_n_1_mot_to_dnc_"></a>
#### all       @ 25-2400/n-1/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2400_var
<a id="seq_0_25___25_2400_n_1_mot_to_dnc_"></a>
#### seq_0_25       @ 25-2400/n-1/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_25_2400_var seq=0,25

<a id="1k_9600___n_1_mot_to_dnc_"></a>
### 1k-9600       @ n-1/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_1_1000_9600_var n_proc=12

<a id="n_2___mot_to_dnc_"></a>
## n-2       @ mot_to_dnc-->dnc_gen
<a id="25_2000_f___n_2_mot_to_dnc_"></a>
### 25-2000-f       @ n-2/mot_to_dnc-->dnc_gen
<a id="all___25_2000_f_n_2_mot_to_dnc_"></a>
#### all       @ 25-2000-f/n-2/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_2_25_2000
<a id="seq_0_25___25_2000_f_n_2_mot_to_dnc_"></a>
#### seq_0_25       @ 25-2000-f/n-2/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_2_25_2000 seq=0,25

<a id="n_3___mot_to_dnc_"></a>
## n-3       @ mot_to_dnc-->dnc_gen
<a id="25_2000_f___n_3_mot_to_dnc_"></a>
### 25-2000-f       @ n-3/mot_to_dnc-->dnc_gen
<a id="all___25_2000_f_n_3_mot_to_dnc_"></a>
#### all       @ 25-2000-f/n-3/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2000
<a id="seq_0_25___25_2000_f_n_3_mot_to_dnc_"></a>
#### seq_0_25       @ 25-2000-f/n-3/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2000 seq=0,25

<a id="25_2400___n_3_mot_to_dnc_"></a>
### 25-2400       @ n-3/mot_to_dnc-->dnc_gen
<a id="all___25_2400_n_3_mot_to_dnc_"></a>
#### all       @ 25-2400/n-3/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2400_var
<a id="seq_0_25___25_2400_n_3_mot_to_dnc_"></a>
#### seq_0_25       @ 25-2400/n-3/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_25_2400_var seq=0,25

<a id="1k_9600___n_3_mot_to_dnc_"></a>
### 1k-9600       @ n-3/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_3_1000_9600_var n_proc=12

<a id="n_5___mot_to_dnc_"></a>
## n-5       @ mot_to_dnc-->dnc_gen
<a id="25_2000_f___n_5_mot_to_dnc_"></a>
### 25-2000-f       @ n-5/mot_to_dnc-->dnc_gen
<a id="all___25_2000_f_n_5_mot_to_dnc_"></a>
#### all       @ 25-2000-f/n-5/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000
<a id="seq_49___25_2000_f_n_5_mot_to_dnc_"></a>
#### seq_49       @ 25-2000-f/n-5/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=49 vis=1
<a id="seq_0___25_2000_f_n_5_mot_to_dnc_"></a>
#### seq_0       @ 25-2000-f/n-5/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=0
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=0 frame_gap=20 win_size=100
<a id="seq_0_25___25_2000_f_n_5_mot_to_dnc_"></a>
#### seq_0_25       @ 25-2000-f/n-5/mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py set=MNIST_MOT_RGB_512x512_5_25_2000 seq=0,25

<a id="detrac___mot_to_dnc_"></a>
## DETRAC       @ mot_to_dnc-->dnc_gen
python3 mot_to_dnc.py seq_set=7 seq=0


