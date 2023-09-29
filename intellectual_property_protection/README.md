# Intellectual Property Protection with ModelGiF

1. Download datasets and pretrained models from [google drive](https://drive.google.com/drive/folders/1idozSeUa9fHQBdPwMGWmQ7GhZuD3Rtpc?usp=sharing)  provided by [Guan et al](https://github.com/guanjiyang/SAC)
   - Including `data`, `adv_train`, `Fine-Pruning`, `finetune_10`, `finetune_model`, and `model`.
2. Generate the reference samples of ModelGiF
   ```shell
   python reference_sample.py
   ```
3. Run the benchmark
   ```shell
   python ipp_task.py -n=field_cosdis -g=0
   ```
    The output results are shown in `run/field_cosdis.log`.
