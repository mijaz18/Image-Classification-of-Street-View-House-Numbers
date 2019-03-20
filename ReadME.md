# Image Captioning with Pytorch

## Usage

Main dependencies: Python 2.7 and Pytorch-0.4.0


#### 1. Download and compile some toolboxes from coco. Please make sure that you are using python 2.7
```bash
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
$ git clone xxxcsc249_hw4
$ cd csc249_hw4
```
#### 2. Download data and evaluation toolboxes
```bash
$ download support.zip from xxx
$ unzip the file and copy folders: data, pycocotoolscap, pycocoevalcap into csc249_hw4
$ pip install -r requirements.txt
$ download stanford NLP models (require java 1.8.0): ./get_stanford_models.sh 
```

#### 3. Train
```bash
$ python train.py
```

#### 4. Prediction for one input image
```bash
$ python predict.py --image img/bird.png
```

#### 5. Evaluation on 100 image from MSCOCO and save the results
```bash
$ python eval.py
```
