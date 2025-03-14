<!-- MarkdownTOC -->

- [n-1](#n_1_)
  - [25-2400       @ n-1](#25_2400___n_1_)
    - [all_seq__6_6       @ 25-2400/n-1](#all_seq_6_6___25_2400_n_1_)
    - [all_seq__6_15       @ 25-2400/n-1](#all_seq_6_15___25_2400_n_1_)
    - [seq_0_25__6_15       @ 25-2400/n-1](#seq_0_25_6_15___25_2400_n_1_)
  - [1k-9600       @ n-1](#1k_9600___n_1_)
    - [seq_0_to_99_win_2-f0_max_4       @ 1k-9600/n-1](#seq_0_to_99_win_2_f0_max_4___1k_9600_n_1_)
    - [seq_0_to_99_win_2-f0_max_16       @ 1k-9600/n-1](#seq_0_to_99_win_2_f0_max_16___1k_9600_n_1_)
      - [best_val_model_178       @ seq_0_to_99_win_2-f0_max_16/1k-9600/n-1](#best_val_model_178___seq_0_to_99_win_2_f0_max_16_1k_9600_n_1_)
  - [100-960       @ n-1](#100_960___n_1_)
    - [live-f0       @ 100-960/n-1](#live_f0___100_960_n_1_)
    - [live-f0-max_4-slide_24       @ 100-960/n-1](#live_f0_max_4_slide_24___100_960_n_1_)
    - [live-f0-max_4-slide_12       @ 100-960/n-1](#live_f0_max_4_slide_12___100_960_n_1_)
      - [mask       @ live-f0-max_4-slide_12/100-960/n-1](#mask___live_f0_max_4_slide_12_100_960_n_1_)
    - [live-f0-max_4-slide_12-64x64-no_repeat       @ 100-960/n-1](#live_f0_max_4_slide_12_64x64_no_repeat___100_960_n_1_)
    - [live-r50-f0-max_4-slide_12-64x64-no_repeat       @ 100-960/n-1](#live_r50_f0_max_4_slide_12_64x64_no_repeat___100_960_n_1_)
    - [live-r50-f0-max_4-slide_24-64x64-no_repeat       @ 100-960/n-1](#live_r50_f0_max_4_slide_24_64x64_no_repeat___100_960_n_1_)
    - [f0_max_4       @ 100-960/n-1](#f0_max_4___100_960_n_1_)
      - [mask       @ f0_max_4/100-960/n-1](#mask___f0_max_4_100_960_n_1_)
    - [f0_max_16       @ 100-960/n-1](#f0_max_16___100_960_n_1_)
      - [mask       @ f0_max_16/100-960/n-1](#mask___f0_max_16_100_960_n_1_)
      - [dnc_to_mot       @ f0_max_16/100-960/n-1](#dnc_to_mot___f0_max_16_100_960_n_1_)
- [n-3](#n_3_)
  - [1k-9600       @ n-3](#1k_9600___n_3_)
    - [all_seq__6_15       @ 1k-9600/n-3](#all_seq_6_15___1k_9600_n_3_)
      - [best_val_model_42       @ all_seq__6_15/1k-9600/n-3](#best_val_model_42___all_seq_6_15_1k_9600_n_3_)
    - [all_seq__6_15       @ 1k-9600/n-3](#all_seq_6_15___1k_9600_n_3__1)
    - [all_seq_15_480_diff_sample       @ 1k-9600/n-3](#all_seq_15_480_diff_sample___1k_9600_n_3_)
      - [best_val_model_6       @ all_seq_15_480_diff_sample/1k-9600/n-3](#best_val_model_6___all_seq_15_480_diff_sample_1k_9600_n_3_)
    - [all_seq_6_480_diff_sample       @ 1k-9600/n-3](#all_seq_6_480_diff_sample___1k_9600_n_3_)
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

<a id="all_seq_6_15___25_2400_n_1_"></a>
### all_seq__6_15       @ 25-2400/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_25_2400_var/all_seq__6_15.cfg --batch_size 32 --cuda

<a id="seq_0_25_6_15___25_2400_n_1_"></a>
### seq_0_25__6_15       @ 25-2400/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_25_2400_var/seq_0_25__6_15.cfg --checkpoint_path log/MNIST_MOT_RGB_512x512_1_25_2400_var/seq_0_25__6_15 --batch_size 32 --cuda

<a id="1k_9600___n_1_"></a>
## 1k-9600       @ n-1-->dnc
<a id="seq_0_to_99_win_2_f0_max_4___1k_9600_n_1_"></a>
### seq_0_to_99_win_2-f0_max_4       @ 1k-9600/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_1000_9600_var/seq_0_to_99_win_2-f0_max_4.cfg --batch_size 16 --num_workers 4

<a id="seq_0_to_99_win_2_f0_max_16___1k_9600_n_1_"></a>
### seq_0_to_99_win_2-f0_max_16       @ 1k-9600/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_1000_9600_var/seq_0_to_99_win_2-f0_max_16.cfg --batch_size 32 --num_workers 0
<a id="best_val_model_178___seq_0_to_99_win_2_f0_max_16_1k_9600_n_1_"></a>
#### best_val_model_178       @ seq_0_to_99_win_2-f0_max_16/1k-9600/n-1-->dnc
CUDA_VISIBLE_DEVICES=0 python3 test.py cfgs_file=cfgs/MNIST_MOT_RGB_512x512_1_1000_9600_var/seq_0_to_99_win_2-f0_max_16.cfg ckpt_name=best_val_model_178.pth max_batches=1

python3 dnc_to_mot.py json=log/MNIST_MOT_RGB_512x512_1_1000_9600_var/seq_0_to_99_win_2-f0_max_16/best_val_model_178_on_validation_/densecap.json set=MNIST_MOT_RGB_512x512_3_1000_9600_var seq=1000

python3 dnc_to_mot.py json=/data/MNIST_MOT_RGB_512x512_1_1000_9600_var/seq_0_to_99_win_2_fix_20.json set=MNIST_MOT_RGB_512x512_1_1000_9600_var seq=1000 @slide size=480 num=2 sample=1

<a id="100_960___n_1_"></a>
## 100-960       @ n-1-->dnc
<a id="live_f0___100_960_n_1_"></a>
### live-f0       @ 100-960/n-1-->dnc
__dbg__
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/seq_0_1_fix_20-live-f0.cfg --batch_size 16 --num_workers 0 --gpu 0

<a id="live_f0_max_4_slide_24___100_960_n_1_"></a>
### live-f0-max_4-slide_24       @ 100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_24-live-f0_max_4.cfg --batch_size 2 --num_workers 4 --distributed 1

<a id="live_f0_max_4_slide_12___100_960_n_1_"></a>
### live-f0-max_4-slide_12       @ 100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-f0_max_4.cfg --batch_size 2 --num_workers 4 --world_size 2 --distributed 1
__dgb__
python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-f0_max_4.cfg --batch_size 2 --num_workers 0 --vis_batch_id 0 --vis_sample_id 0

<a id="mask___live_f0_max_4_slide_12_100_960_n_1_"></a>
#### mask       @ live-f0-max_4-slide_12/100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-f0_max_4-mask.cfg --batch_size 2 --num_workers 4 --world_size 2 --distributed 1

__dgb__
python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-f0_max_4.cfg --batch_size 2 --num_workers 0

python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/seq_0_1_slide_12-live-f0_max_4.cfg --batch_size 2 --num_workers 0 --gpu 0

<a id="live_f0_max_4_slide_12_64x64_no_repeat___100_960_n_1_"></a>
### live-f0-max_4-slide_12-64x64-no_repeat       @ 100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-f0_max_4-64x64-no_repeat.cfg --batch_size 2 --num_workers 4 --world_size 2 --distributed 1 --sent_weight 0.5 --vis_id=0

python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-f0_max_4-64x64-no_repeat.cfg --batch_size 2 --num_workers 4 --world_size 2 --distributed 0 --vis_id=0

<a id="live_r50_f0_max_4_slide_12_64x64_no_repeat___100_960_n_1_"></a>
### live-r50-f0-max_4-slide_12-64x64-no_repeat       @ 100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 16 --num_workers 4 --world_size 2 --distributed 1 --sent_weight 0.5

python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 16 --num_workers 4 --world_size 2 --distributed 0 --vis_id=0

python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_12-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 16 --num_workers 4 --world_size 2 --distributed 0 --vis_id=0

<a id="live_r50_f0_max_4_slide_24_64x64_no_repeat___100_960_n_1_"></a>
### live-r50-f0-max_4-slide_24-64x64-no_repeat       @ 100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_24-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 8 --num_workers 4 --world_size 2 --distributed 1 --sent_weight 0.5

python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_slide_24-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 8 --num_workers 4 --world_size 2 --distributed 0 --vis_id=0
__dgb__
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/seq_0_5_slide_24-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 4 --num_workers 4 --world_size 2 --distributed 1 --sent_weight 0.75

python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/seq_0_5_slide_24-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 8 --num_workers 4 --world_size 2 --distributed 0 --vis_id=0

python train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/seq_0_1_slide_16-live-r50-f0_max_4-64x64-no_repeat.cfg --batch_size 8 --num_workers 4 --world_size 2 --distributed 0 --vis_id=0


<a id="f0_max_4___100_960_n_1_"></a>
### f0_max_4       @ 100-960/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_fix_20-f0_max_4.cfg --batch_size 16 --num_workers 0
<a id="mask___f0_max_4_100_960_n_1_"></a>
#### mask       @ f0_max_4/100-960/n-1-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_fix_20-f0_max_4-mask.cfg --batch_size 8 --num_workers 0

<a id="f0_max_16___100_960_n_1_"></a>
### f0_max_16       @ 100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_fix_20-f0_max_16.cfg --batch_size 24 --num_workers 4 --world_size 2 --distributed 1
__dbg__
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/seq_0_1_fix_20-f0_max_16.cfg --batch_size 32 --num_workers 0
<a id="mask___f0_max_16_100_960_n_1_"></a>
#### mask       @ f0_max_16/100-960/n-1-->dnc
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_fix_20-f0_max_16-mask.cfg --batch_size 24 --num_workers 4 --world_size 2 --distributed 1

<a id="dnc_to_mot___f0_max_16_100_960_n_1_"></a>
#### dnc_to_mot       @ f0_max_16/100-960/n-1-->dnc
python3 dnc_to_mot.py json=/data/MNIST_MOT_RGB_512x512_1_100_960_var/all_seq_fix_20.json set=MNIST_MOT_RGB_512x512_1_100_960_var seq=100 @slide size=480 sample=1

<a id="n_3_"></a>
# n-3
<a id="1k_9600___n_3_"></a>
## 1k-9600       @ n-3-->dnc
<a id="all_seq_6_15___1k_9600_n_3_"></a>
### all_seq__6_15       @ 1k-9600/n-3-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq__6_15.cfg --batch_size 32 --num_workers 4
<a id="best_val_model_42___all_seq_6_15_1k_9600_n_3_"></a>
#### best_val_model_42       @ all_seq__6_15/1k-9600/n-3-->dnc
CUDA_VISIBLE_DEVICES=0 python3 test.py cfgs_file=cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq__6_15.cfg ckpt_name=best_val_model_42.pth max_batches=1

python3 dnc_to_mot.py json=log/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq__6_15/epoch_88_on_validation_ set=MNIST_MOT_RGB_512x512_3_1000_9600_var seq=1000


python3 dnc_to_mot.py json=/data/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq.json set=MNIST_MOT_RGB_512x512_3_1000_9600_var seq=1000

python3 dnc_to_mot.py json=/data/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq_15_480_diff_sample.json set=MNIST_MOT_RGB_512x512_3_1000_9600_var seq=1000 vocab_fmt=1 max_diff=99 @slide sample=15 size=480

python3 dnc_to_mot.py json=/data/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq_15_480_diff.json set=MNIST_MOT_RGB_512x512_3_1000_9600_var seq=1000 vocab_fmt=1 max_diff=99 @slide sample=15 size=480

<a id="all_seq_6_15___1k_9600_n_3__1"></a>
### all_seq__6_15       @ 1k-9600/n-3-->dnc
python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq__6_15.cfg --batch_size 32 --num_workers 4

<a id="all_seq_15_480_diff_sample___1k_9600_n_3_"></a>
### all_seq_15_480_diff_sample       @ 1k-9600/n-3-->dnc
CUDA_VISIBLE_DEVICES=0 python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq_15_480_diff_sample.cfg --batch_size 32 --num_workers 4 --n_proc 12
<a id="best_val_model_6___all_seq_15_480_diff_sample_1k_9600_n_3_"></a>
#### best_val_model_6       @ all_seq_15_480_diff_sample/1k-9600/n-3-->dnc
CUDA_VISIBLE_DEVICES=0 python3 test.py cfgs_file=cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq_15_480_diff_sample.cfg ckpt_name=best_val_model_6.pth max_batches=100

python3 dnc_to_mot.py json=log/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq_15_480_diff_sample/best_val_model_6_on_validation_ set=MNIST_MOT_RGB_512x512_3_1000_9600_var seq=1000 @ vocab_fmt=1 max_diff=99 @slide sample=15 size=480

<a id="all_seq_6_480_diff_sample___1k_9600_n_3_"></a>
### all_seq_6_480_diff_sample       @ 1k-9600/n-3-->dnc
CUDA_VISIBLE_DEVICES=1 python3 train.py --cfgs_file cfgs/MNIST_MOT_RGB_512x512_3_1000_9600_var/all_seq_6_480_diff_sample.cfg --batch_size 32 --num_workers 4 --n_proc 12

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
























































































