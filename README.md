# Tensorflow Implementation of Error Encoding Network
This is a tensorflow implementation of EEN developed by Mikael Henaff, Junbo Zhao and Yann LeCun from Facebook AI Research, Courant Institute, New York University. Original work can be found in the link below.<br>
[Click Here to View Original Work](https://github.com/mbhenaff/EEN
)
<br>
<br>
Developed by James Youngchae Chee @ Dongsin Science Highschool.
![Diagram](img/een-crop.png)
## Usage - 한국어(Korean)
#### 파일 구조

	Project Directory
	    ├ models.py
	    ├ dataloader.py
	    ├ train_een_deterministic.py
	    ├ train_een_latent.py
	    ├── data (새로 만들어야됨 --> 학습시킬 영상들과 tfrecord가 저장될 공간)
	    │   ├── flower.mp4 (학습 시킬 영상들 반드시 필요)
	    │	├── dataset.tfrecords (없을 경우 학습코드를 실행시키면 생성됨)
	    │   └── ...
	    └── model (새로 만들어야됨 --> 모델들이 저장될 공간)

### 1. Data
리포지토리를 클론 한 후 ```./data``` 디렉토리를 만들어 원하는 영상 파일을 넣는다.
### 2. Training
#### Deterministic 모델 학습시키기
```
Terminal> python3 train_een_deterministic.py [옵션 설정]
```
#### Latent 적용된 모델 학습시키기
```
Terminal> python3 train_een_latent.py [옵션 설정]
```
#### 반드시 설정할 옵션
1. ``` -videopath ``` : 비디오가 저장된 디렉토리
2. ``` -tfrecordspath ``` : tfrecords 파일의 디렉토리 (존재하지 않는다면 설정한 디렉토리로 자동 생성됨)
3. ```-model_name``` (only deterministic 모델) : 학습한 모델을 저장할 디렉토리 및 파일명 (기본값: './model/deterministic/deterministic_model')
4. ```-model_path``` (only latent 모델) : deterministic 모델을 불러올 디렉토리 및 파일명 (기본값: './model/deterministic/deterministic_model-19.meta')
5. ``` -save_dir ``` : 학습한 모델을 저장할 디렉토리 (기본값: './results/')

#### 이외의 옵션들
1. ``` -width ``` : 학습할 때 사용할 프레임의 너비(픽셀단위) (기본값: 480)
2. ``` -height ``` : 학습할 때 사용할 프레임의 높이(픽셀단위) (기본값: 480)
3. ``` -pred_frame ``` : 예측할(입력할) 프레임 수 (기본값: 5)
4. ``` -time_interval ``` : 예측한(입력한) 프레임들 간 시간 간격(milisecond) (기본값: 2)
5. ``` -data_interval ``` : 영상에서 프레임을 추출해 학습데이터를 만들때 시작 프레임간 간격 (프레임단위) (기본값: 150)
6. ``` -batch_size ``` : 학습시 배치 크기 (기본값: 5)
7. ``` -nfeature ``` : Conv net 구조에서의 feature 수 (기본값: 64)
8. ``` -nlatent ``` : Latent 값의 갯수 (기본값: 4) ** train_een_latent.py에 만 존재**
9. ``` -lrt ``` : learning rate (기본값: 0.0005)
10. ``` -epoch ``` : epoch 수 (기본값: 500)

### 3. Visualization
<i>개발중</i>
## Usage - English
#### Required Directory Structure

	Project Directory
	    ├ models.py
	    ├ dataloader.py
	    ├ train_een_deterministic.py
	    ├ train_een_latent.py
	    ├── data (need to be created --> where videos and tfrecord are stored)
	    │   ├── flower.mp4 (Need to add at least one video to train)
	    │	├── dataset.tfrecords (auto created if not exist)
	    │   └── ...
	    └── model (need to be created --> where trained deterministic models are stored)

### 1. Data
Clone this repository and make ```./data``` directory. Add video you want to train.
### 2. Training
#### Training Deterministic Model
```
Terminal> python3 train_een_deterministic.py [initialize options]
```
#### Training Latent Residual Model
```
Terminal> python3 train_een_latent.py [initialize options]
```
#### Crucial Options to Check
1. ``` -videopath ``` : directory of training video
2. ``` -tfrecordspath ``` : tfrecords file directory (auto created if not exist)
3. ```-model_name``` (only deterministic model) : deterministic model path (default: './model/deterministic/deterministic_model')
4. ```-model_path``` (only latent model) : latent model path (default: './model/deterministic/deterministic_model-19.meta')
5. ``` -save_dir ``` : where models are saved (default: './results/')

#### More options to go through
1. ``` -width ``` : wanted width for training(pixel) (default: 480)
2. ``` -height ``` : wanted height for training(pixel) (default: 480)
3. ``` -pred_frame ``` : number of frames to learn and predict (default: 5)
4. ``` -time_interval ``` : time interval between frames in milliseconds (default: 2)
5. ``` -data_interval ``` : number of frame interval between start of each dataset (default: 150)
6. ``` -batch_size ``` : batch size (default: 5)
7. ``` -nfeature ``` : number of feature maps in convnet (default: 64)
8. ``` -nlatent ``` : Number of Latent Variables (default: 4) ** train_een_latent.py ONLY**
9. ``` -lrt ``` : learning rate (default: 0.0005)
10. ``` -epoch ``` : number of training epoch (default: 500)

### 3. Visualization
<i>In Development</i>
