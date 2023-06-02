# MusicVAE (Pytorch)
Music VAE : https://arxiv.org/pdf/1803.05428.pdf
Groove MIDI Dataset: https://magenta.tensorflow.org/datasets/groove

#### 논문 요약
<p align="center">
<img width="519" alt="image" src="https://github.com/EJueon/music_VAE/assets/93572176/44233bdf-8717-4471-bad1-3cbab582d77c">
</p>

- RNN 계열의 신경망의 근본적인 한계인 long-term structure에 대해서 비교적 낮은 성능을 보인다는 문제(posterior collapse)를 해결하기 위하여 제안
- 기본적인 RNN 기반 Encoder-Decoder(Seq2Seq)이 아니라 Variational Autoencoder(VAE)를 사용하였음, ELBO loss 사용
- 인코더에서 생성하는 latent representation을 그대로 사용하는 것이 아니라, 전체 시퀀스를 기준으로 부분 시퀀스에 대한 embedding 생성. 디코더에 해당 embedding과 입력값을 입력하는 것으로 전체 latent representation을 그대로 사용하는 것보다 posterior collapse 문제를 다소 회피할 수 있음
- 기존 베이스라인 모델격인 flat보다 전반적으로 높은 성능을 보여주었음 
<br /> <br /> <br /> 





## How to use
#### preprocess
```shell
python main.py --opt=preprocess --config=../config path
```

#### train
```shell
python main.py --opt=train --config=../config path
```
#### generate
```shell
python main.py --opt=generate --config=../config path --model_path=../data/ckpt/epoch_100.pt --file_path=../dataset/test.midi
```

<br /> <br /> <br /> 

## 과제 개요
- **주제** : Music VAE, Groove MIDI Dataset를 사용하여 전처리-학습 과정을 거쳐 4마디에 해당하는 드럼 샘플 추출
- **기간** : 23.5.31 10:00 ~ 23.06.01 10:00

- **프로젝트 구조**
        
    ```bash
        ├─ configs
        │  └─ basic_config.yaml
        ├─ src
        │  ├─ main.py
        │  ├─ models
        │  │  ├─ dataloader.py
        │  │  ├─ metrics.py
        │  │  └─ musicVAE.py
        │  └─ utils
        │     ├─ __init__.py
        │     ├─ process_utils.py
        │     └─ trainer.py
        ├─ data
        ├─ requirements.txt
        └─ .gitignore 
    ```
        
- **데이터셋 구조**
        
    ```jsx
        filepath : 전처리시 MIDI 데이터의 파일명 및 경로
        data : drum beats만을 추출하여 시퀀스화된 MIDI 데이터 (2**drum_classes)
    ```
<br /> <br /> <br /> 
## 과제 진행사항 

> ## 📌 
> - Preprocessing : 4/4 기준 드럼 pitches 전처리
> - Training : MusicVAE 모델 및 학습 평가 모듈 구현
> - Generate : (TODO) 4 마디 기준 MIDI 생성
    
### Preprocessing

- pretty_midi 모듈을 통해 MIDI 파일로 부터 데이터 추출
- pitches를 9개의 대표 클래스로 매핑하여 양자화 
- 논문과 동일하게 2^9(512)개의 유형으로 변환 
- 양자화된 MIDI 데이터 저장 


### Model Structure
#### Bidirectional encoder 
- 2-layer Bidirectional LSTM encoder
- hidden state를 토대로 mu와 sigma 인코딩, 각각을 인코딩하기 위해서 linear layer 사용 (Eq.6, Eq.7)
- z = mu + (sigma * epsilon) (Eq.2)로 latent representation z 인코딩. 
- z는 embedding vector를 연산하기 용이하도록 차원 변경 (batch_size, latent_dim)  -> (batch_size, self.U, self.input_size)       

#### Hierarchical decoder 
- u ∈ U, U는 부분 시퀀스의 개수일 때, 각 z[:, u, :]에 대해서 embedding 값으로 변환
- embedding과 이에 해당하는 입력 데이터의 부분 시퀀스를 함께 디코더에 입력


#### Training 
- sequence length = 64 (4 bar)
- learning rate = 1e-3
- AdamW optimizer 
- CosineAnnealingWarmRestarts scheduler
- criterion = ELBO loss
- train dataset과 dev dataset은 8:2로 고정하여 학습 진행 

<img width="967" alt="" src="https://github.com/EJueon/musicVAE/assets/93572176/41131e3f-36d1-4f6a-a676-87ada2acc8d3">
