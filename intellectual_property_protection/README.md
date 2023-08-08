# Intellectual Property Protection with ModelGiF

1. Download datasets and pretrained models from [google drive](https://drive.google.com/drive/folders/1idozSeUa9fHQBdPwMGWmQ7GhZuD3Rtpc?usp=sharing)  provided by [this work](https://github.com/guanjiyang/SAC)

   - including data, adv_train, Fine-Pruning, finetune_10, finetune_model, model.

2. Generate the reference samples of ModelGiF

   ```shell
   python reference_sample.py
   ```

3. Run the benchmark

   ```shell
   python ipp_task.py -n=field_cosdis -g=0
   ```


