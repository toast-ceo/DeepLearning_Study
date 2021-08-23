# 딥러닝 Numpy 공부

# 유틸 im2col col2im & 데이터 로더

## im2col

- 다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화)

    ![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled.png)

```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = cp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    #print(col)
    return col
```

## col2im

- 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
- 평탄화 작업과 반대되는 개념

```python
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = cp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
```

## 소프트맥스 함수 (Softmax Funtion)

- 세 개 이상으로 분류하는 다중 클래스 분류에서 사용되는 활성화 함수
- 분류될 클래스가 n개라고 할 때, n차원의 벡터를 입력받아, 각 클래스에 속할 확률을 추정함

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%201.png)

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%202.png)

- 시그모이드 함수처럼 출력층에서 주로 사용, 이진 분류에서만 사요오디는 시그모이드 함수와 달리 다중 불류에서 주료 사용됨.
- 확률의 총합이 1이므로,  어떤 분류에 속할 확률이 가장 높을지를 쉽게 인지할 수 있다.

```python
def softmax(x):
    if x.ndim == 2: #2차원일 때
        x = x.T # .T 전치 함수
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T 

    x = x - cp.max(x) # 오버플로 대책
    return cp.exp(x) / cp.sum(cp.exp(x))
```

# 손실함수

- '하나의 지표'를 기준으로 최적의 매개변수 값을 탐색한다고 할때, 신경망 학습에서 사용하는 지표를 손실 함수라고 한다.
- 사용하는 이유? 최적의 매개변수를 탐색할 때 손실 함수의 값을 가능한 한 작게 하는 매개변수 값을 찾음. 
손실 함수를 이용하면 매개변수가 바뀔 때 해당 값이 연속적으로 바뀌게 된다. (ex) 0.932123 ⇒ 0.945342이런 식으로)
정확도를 지표로 삼아버리면 매개변수의 미분이 대부분의 장소에서 0이 됨. (ex) 33% , 32%)

## mean_squared_error (오차제곱합)

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%203.png)

- 가장 많이 쓰이는 손실 함수
- yi는 신경망의 출력, ti는 정답 레이블, n은 데이터의 차원 수를 나타냄

## cross_entropy_error (교차 엔트로피 오차)

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%204.png)

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%205.png)

- 손실함수

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%206.png)

- 자연 로그의 그래프
- 원-핫 인코딩 방식: 정답에 해당하는 인덱스의 원소만 1이고 나머지는 0

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    #print("t.reshape=> ", t)            
    #print("y.reshape=> ", y)            
    batch_size = y.shape[0]
    #print("batch_size = y.shape[0]=> ", batch_size)
    
    #return -cp.sum(cp.log(y[cp.arange(batch_size), t])) / batch_size
    return -cp.sum(cp.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

## +) MNIST 데이터 셋 읽기

- mnist란? 
: 손으로 쓴 숫자들로 이루어진 대형 데이터베이스이며, 다양한 화상 처리 시스템을 트레이닝하기 위해 일반적으로 사용된다.

```python
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(cp.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
```

# 레이어 테스트

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%207.png)

## ReLU

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%208.png)

- 활성화 함수 계층

```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```

- mask라는 인스턴스 변수를 가짐
- 순전파의 입력인 x의 원소 값이 0 이하인 인덱스는 True, 그 외 (0보다 큰원소)는 False로 유지

## Dropout

- 오버피팅을 억제하는 방식
- 뉴런을 임의로 삭제하면서 학습하는 방법

```python
class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = cp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
```

## Affine

- 행렬의 계산은 차원 등 신경쓸게 많은데, 이런 행렬의 곱을 보통 Affine이라 부른다.
- ANN에서 Affine이란, 기족의 단일 값, 혹은 단일 차원 배열로 넘겨주던 입력값을 행렬로서 받아드려 한번에 처리하는 것.
- Affine구조란, 그런 행렬을 입력값으로 받아와 순전파와 역전파를 구현할 수 있는 구조
- 밑 코드의 경우 완전 연결된 계층이라고 표현 할 수도 있다
- 완전연결 계층으로 이루어진 네트워크는 Affine-Relu로 구현
- [https://dsbook.tistory.com/59](https://dsbook.tistory.com/59) 참고

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = cp.dot(self.x, self.W) + self.b
     

    def backward(self, dout):
        dx = cp.dot(dout, self.W.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
```

⇒ 정확한 내용은 질문 

## SoftmaxWithLoss

- Softmax 함수와 cross_entropy_error함수가 합쳐진 계층
- 출력층에서 사용됨

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[cp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        #print("dx.shape => ", dx.shape)
        #print("SoftmaxWithLoss backward: dx => ", dx)
        return dx
```

## Convolution

- 이 부분에 대해선 따로 정리할 예정

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        #print("컨볼루션 순전파")
        FN, C, FH, FW = self.W.shape
        #print("FN, C, FH, FW =>",FN, C, FH, FW)
        
        N, C, H, W = x.shape
        #print("N, C, H, W =>",N, C, H, W)
        
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = cp.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        #print("컨볼루션 역전파")
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = cp.sum(dout, axis=0)
        self.dW = cp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = cp.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
```

## BatchNormalization (배치 정규화)

### Gradient Vanishing / Exploding 문제

- **Gradient 라는 것이 결국 미분값 즉 변화량을 의미하는데 이 변화량이 매우 작아지거나(Vanishing) 커진다면(Exploding) 신경망을 효과적으로 학습시키지 못하고, Error rate 가 낮아지지 않고 수렴해버리는 문제가 발생**하게 된다.
- 이러한 문제를 해결하기 위해 활성화 함수들은 비선형적인 방식으로 입력 값을 매우 작은 출력 값의 범위로 squash해버린다
    - 이렇게 출력의 범위를 설정할 경우 매우 넓은 입력 값의 범위가 극도로 작은 범위의 결과값으로 매핑 됨
    - 첫 레이어의 입력 값에 대해 매우 큰 변화량이 있더라도 결과 값의 변화량은 극소가 되버린다
- 이러한 문제를 해결하기 위해
    1. **Change activation function** : 활성화 함수 중 Sigmoid 에서 이 문제가 발생하기 때문에 ReLU 를 사용
    2. **Careful initialization** : 가중치 초기화를 잘 하는 것을 의미
    3. **Small learning rate** : Gradient Exploding 문제를 해결하기 위해 learning rate 값을 작게 설정함
- 위 와 같은 방법이 있음에도 학습하는 과정 자체를 전체적으로 안정화"하여 학습 속도를 가속 시킬 수 있는 근본적인 방법인 "배치 정규화(Batch Normalization)를 쓰는 것이 좋다
- 기본적으로 정규화를 하는 이유 ⇒ 학습을 더 빠르게, Local optimum 문제에 빠지는 가능성을 줄이기 위해
- **단순하게 Whitening만을 시킨다면 이 과정과 파라미터를 계산하기 위한 최적화(Backpropagation)과 무관하게 진행되기 때문에 특정 파라미터가 계속 커지는 상태로 Whitening 이 진행 될 수 있다.** Whitening 을통해 손실(Loss)이 변하지 않게 되면, 최적화 과정을 거치면서 특정 변수가 계속 커지는 현상이 발생할 수 있다.

⇒ 위와 같은 이유의 문제점을 해결하도록 한 트릭이 배치 정규화 

- 각 레이어마다 정규화 하는 레이어를 두어, 변형된 분포가 나오지 않도록 조절하게 하는 것이 배치 정규화이다.

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%209.png)

- 배치 정규화는 간단히 말하자면 미니배치의 평균과 분산을 이용해서 정규화 한 뒤에, scale 및 shift 를 감마(γ) 값, 베타(β) 값을 통해 실행한다. 이 때 감마와 베타 값은 학습 가능한 변수이다. 즉, 역전파를 통해서 학습이 된다.

```python
class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = cp.zeros(D)
            self.running_var = cp.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = cp.mean(xc**2, axis=0)
            std = cp.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((cp.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = cp.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -cp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = cp.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
```

## Pooling

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%2010.png)

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20Numpy%20%E1%84%80%E1%85%A9%E1%86%BC%E1%84%87%E1%85%AE%20fe753b40fb7942ca93cd43640a7dfbe2/Untitled%2011.png)

- Convolution을 거쳐서 나온 activation maps이 있을 때,이를 이루는 convolution layer을 resizing하여 새로운 layer를 얻는 것

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        #print("풀링 순전파")
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        #print(out.shape)
        return out

    def backward(self, dout):
        #print("풀링 역전파")
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = cp.zeros((dout.size, pool_size))
        dmax[cp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
```