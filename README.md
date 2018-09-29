# Tensorflow Implementation of Error Encoding Network
This is a tensorflow implementation of EEN developed by Mikael Henaff, Junbo Zhao and Yann LeCun from Facebook AI Research, Courant Institute, New York University. Original work can be found in the link below.<br>
[Click Here to View Original Work](https://github.com/mbhenaff/EEN
)
<br>
<br>
Developed by James Youngchae Chee @ Dongsin Science Highschool.
![Diagram](img/een-crop.png)
## Usage - 한국어(Korean)
#### Deterministic 모델 학습시키기
```
Terminal> python3 train_een_deterministic.py [옵션 설정]
```
#### Latent 적용된 모델 학습시키기
<i>개발중</i>
```
Terminal> python3 train_een_latent.py [옵션 설정]
```
### 1. Data
리포지토리를 클론 한 후 ```./data``` 디렉토리를 만들어 원하는 영상 파일을 넣는다.
### 2. Training
#### 반드시 설정할 옵션
1. ``` -videopath ``` : 비디오가 저장된 디렉토리
2. ``` -tfrecordspath ``` : tfrecords 파일의 디렉토리 (존재하지 않는다면 설정한 디렉토리로 자동 생성됨)
3. ```-model_name``` (only deterministic 모델) : 학습한 모델을 저장할 디렉토리 및 파일명 (기본값: './model/deterministic/deterministic_model')
4. ```-model_path``` (only latent 모델) : deterministic 모델을 불러올 디렉토리 및 파일명 (기본값: './model/deterministic/deterministic_model-19.meta')
5. ``` -save_dir ``` : 학습한 모델을 저장할 디렉토리

#### 이외의 옵션들
1. ``` -width ``` : 학습할 때 사용할 프레임의 너비(픽셀단위) (기본값: 480)
2. ``` -height ``` : 학습할 때 사용할 프레임의 높이(픽셀단위) (기본값: 480)
3. ``` -pred_frame ``` : 예측할(입력할) 프레임 수 (기본값: 5)
4. ``` -time_interval ``` : 예측한(입력한) 프레임들 간 시간 간격(milisecond) (기본값: 2)
5. ``` -frame_interval ``` : 영상에서 프레임을 추출해 학습데이터를 만들때 시작 프레임간 간격 (프레임단위) (기본값: 150)
6. ``` -batch_size ``` : 학습시 배치 크기 (기본값: 5)
7. ``` -nfeature ``` : Conv net 구조에서의 feature 수 (기본값: 64)
8. ``` -lrt ``` : learning rate (기본값: 0.0005)
9. ``` -epoch ``` : epoch 수 (기본값: 500)

### 3. Visualization
<i>개발중</i>
## Usage - English
### Data
### Training
### Visualization
