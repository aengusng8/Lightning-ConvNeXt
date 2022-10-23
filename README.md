<p align="center">
  <img src="https://user-images.githubusercontent.com/67547213/197374538-6ae342ff-abde-48fd-9603-afacbdfe304a.png">
</p>

-------------

### Training
```bash
CUDA_VISIBLE_DEVICES="1,2,3" python3 main.py --epoch 15 --batch_size 1 --nb_classes 20 --data_path /home/anhducnguyen/pascal_2007 --train_csv_path /home/anhducnguyen/pascal_2007/multihot_train.csv --test_csv_path /home/anhducnguyen/pascal_2007/multihot_test.csv
```
