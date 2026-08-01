Some experiments using YOLO

### Installation

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Train

* Configure your dataset path in `main.py` for training
* Run `python main.py --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results


### BDD Dataset structure

    ├── bdd 
        ├── images
            ├── 100k
                ├── train
                    ├── 1111.jpg
                    ├── 2222.jpg
                ├── test
                    ├── 1111.jpg
                    ├── 2222.jpg
                ├── val
                    ├── 1111.jpg
                    ├── 2222.jpg
        ├── labels
            ├── bdd100k_labels_images_train.json
            ├── bdd100k_labels_images_test.json
            ├── bdd100k_labels_images_val.json

#### Reference

* https://github.com/jahongir7174/YOLOv8-pt
