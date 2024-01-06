python3 -m spacy download en

# anet

python3 test.py gpu=0 densecap_eval_file=./tools/densevid_eval/evaluate.py batch_size=1 start_from=./checkpoint/anet-2L-gt-mask/model_epoch_19.t7 id=anet-2L-gt-mask-19 val_data_folder=validation learn_mask=1 gated_mask=1 cuda=1

# yc2

python3 test.py gpu=0 densecap_eval_file=./tools/densevid_eval/evaluate.py batch_size=1 start_from=./checkpoint/yc2-2L-gt-mask/model_epoch_19.t7 id=yc2-2L-gt-mask-19 val_data_folder=validation learn_mask=1 gated_mask=1





































