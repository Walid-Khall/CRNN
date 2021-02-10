# Scanned Receipt OCR by Convolutional-Recurrent Neural Network

This is a `pytorch` implementation of CRNN


## Dependency

1. pytorch
2. lmdb

## Prediction

CRNN only accepts image that has single line of word. Therefore, we must provide bounding boxes for the whole reciept image.

1. Put image under folder `./data_test/` and bounding box text file under `./boundingbox/`, the name of image file and text file must correspond. 

2. To predict, just run `python main.py`. You can change the code inside to visualise output or prepare result for task 3.

example result:
```
tan chay yee
81750 masai johor
sales persor : fatin
tax invoice
total inclusive gst:
invoice no : pegiv1030765
email: ng@ojcgroup.com
bill to"
date
description
address
total:
cashier
:the peak quarry works
```

## Training

Training a CRNN requires converting data into `lmdb` format.

1. Divide training data into training and validating dataset, put each portion into `./data_tain/` and `./data_valid`.

2. Run `create_dataset.py` to create lmdb dataset. The created dataset could be found inside `./dataset`

3. After preparing dataset, just run:
   ```shell
   python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda
   ```
   with desired options

4. Trained model output will be in `./expr/`