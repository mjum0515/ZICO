from __future__ import annotations # 코드 정리용

from dataclasses import dataclass # LSTM 상태 관리 단순화

import torch # 텐서 연산, 입력, 가중치, 출력 처리
from torch import nn # 모델 레이어 구성

"""
<안 된 것>
1. 파일에서 데이터 읽기
2. 윈도우 샘플 생성
3. DataLoader 배치화
4. loss 계산
5. optimizer.step()
6. 학습
7. validation
8. 결과 저장

<된 것>
1. 입력이 어떤 모양이어야 하는지 정의
2. 모델이 그 입력을 받아 어떤 출력을 내야 하는지 정의
3. GPU/MPS에서 forward 계산이 가능한지 확인

입력: 과거 t-2:t의 ocean, 과거 t-2:t의 atmos
출력: 미래 t+1:t+L의 ΔT, ΔS, ΔU, ΔV

<현재 파일의 역할>
1. ocean 입력과 atmos 입력을 받는다
2. 둘을 채널 방향으로 합친다
3. ConvLSTM으로 시간축을 읽는다
4. 마지막 hidden state를 뽑는다
5. CNN head로 미래 ΔT, ΔS, ΔU, ΔV를 낸다

<현재 파일 기준 설정>
입력 길이: 3 시점
ocean 채널: 4(T, S, U, V)
atmos 채널: 5(u10, v10, t2m, q2m, rsns)
출력 채널: 4(ΔT, ΔS, ΔU, ΔV)
리드타임: 3(t+1, t+2, t+3)

A1: ECCO ocean (t-2:t) + ERA5 atmos (t-2:t) -> ΔX(t+1:t+3)
A3: CMIP6 ocean (t-2:t) + CMIP6 atmos (t-2:t) -> ΔX(t+1:t+3)

<현재 버전>
Exp 1: Forecast mode, SSH 제외, A1/A3 공통 backbone
"""


@dataclass(frozen=True)
class ConvLSTMState: # ConvLSTM의 내부 상태인 hidden, cell 하나의 구조로 묶기
    # h: hidden state, c: cell state
    h: torch.Tensor # 지금까지 본 정보 요약
    c: torch.Tensor # LSTM이 장기 기억처럼 유지하는 값


class ConvLSTMCell(nn.Module):
    """
    [구조 순서 1]
    ConvLSTM의 최소 계산 단위.

    한 시점의 입력 x_t와 이전 시점의 기억(state.h, state.c)을 받아
    다음 시점의 기억(h_next, c_next)을 만든다.

    일반 LSTM은 선형층을 쓰지만,
    ConvLSTM은 공간 구조(lat, lon)를 유지하기 위해 Conv2d로 gate를 계산한다.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.") # 홀수 커널만 허용(공간 크기 유지)

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels, # 현재 입력(x_t), 이전 hidden state(h_{t-1})를 붙여 convolution 한 번 수행
            4 * hidden_channels, # LSTM gate가 4개(input, forget, candidate, output)
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_state( # 시퀀스 시작 시 hidden state와 cell state를 0으로 만듦
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ConvLSTMState:
        zeros = torch.zeros( # 0에서 시작(이전 기억 없음)
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        return ConvLSTMState(h=zeros, c=zeros.clone()) # h, c는 같은 공간 shape

    def forward(self, x_t: torch.Tensor, state: ConvLSTMState) -> ConvLSTMState:
        combined = torch.cat([x_t, state.h], dim=1) # 현재 입력과 이전 hidden state를 채널 방향으로 붙임
        gates = self.gates(combined)
        i, f, g, o = torch.chunk(gates, chunks=4, dim=1) # convolution 결과를 4개 gate로 나눔

        i = torch.sigmoid(i)
        f = torch.sigmoid(f) # 0~1
        g = torch.tanh(g) # -1~1 (gate 값 범위 조절)
        o = torch.sigmoid(o)

        c_next = f * state.c + i * g # 이전 기억을 얼마나 유지하고 새 정보를 얼마나 반영할지 결정
        h_next = o * torch.tanh(c_next) # 최종 hidden state 계산
        return ConvLSTMState(h=h_next, c=c_next)

# LSTM의 표준 갱신식을 convolution 기반 gate 계산과 결합해, 공간 구조를 유지한 채 시계열 기억을 업데이트합니다.


class ConvLSTM(nn.Module):
    """
    [구조 순서 2]
    여러 시점의 지도를 시간 순서대로 읽는 backbone.

    입력 shape: [batch, time, channel, lat, lon]

    예를 들어 t-2, t-1, t 세 시점이 들어오면,
    각 시점을 순서대로 읽으면서 시공간 정보를 hidden state에 누적한다.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: tuple[int, ...] = (32, 32), # ConvLSTM 층을 2개 쌓겠다
        kernel_size: int = 3,
    ):
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels must not be empty.")

        channels = [input_channels, *hidden_channels]
        self.layers = nn.ModuleList( # 여러 층을 리스트처럼 관리
            [
                ConvLSTMCell(
                    input_channels=channels[i],
                    hidden_channels=channels[i + 1],
                    kernel_size=kernel_size,
                )
                for i in range(len(hidden_channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5: # 입력이 [B, T, C, H, W] 형식인지 검사
            raise ValueError(f"x must be [B, T, C, H, W], got {tuple(x.shape)}")

        batch_size, seq_len, _, height, width = x.shape
        current = x # 현재 처리할 시퀀스 준비

        # 여러 ConvLSTM 층을 순서대로 통과시킨다.
        for layer in self.layers:
            state = layer.init_state( # 각 층 시작 시 hidden/cell state를 0으로 초기화
                batch_size=batch_size,
                height=height,
                width=width,
                device=x.device,
                dtype=x.dtype,
            )
            outputs = []

            # 시간축을 따라 t=0,1,2,... 순서로 읽으면서
            # hidden state를 계속 업데이트한다.
            for t in range(seq_len):
                state = layer(current[:, t], state)
                outputs.append(state.h)

            # 한 층의 모든 시점 출력은 다음 층의 입력 시퀀스가 된다.
            current = torch.stack(outputs, dim=1)

        return current

# 각 층에서 입력 시퀀스를 시간 순서대로 처리하고, 한 층의 hidden sequence를 다음 층 입력으로 전달하는 다층 ConvLSTM 구조

# 최종 예측 모델은 OceanConvLSTMForecast이며, 해양 상태장과 대기 forcing을 함께 입력받아 미래 해양 변화량을 출력

class OceanConvLSTMForecast(nn.Module):
    """
    [구조 순서 3: 메인 모델]

    입력:
      - x_ocean: [B, T, 4, H, W]
      - x_atmos: [B, T, 5, H, W]

    출력:
      - y_hat: [B, L, 4, H, W]

    예:
      과거 t-2:t의 해양/대기장을 넣고
      미래 t+1:t+L의 해양 변화량 ΔT, ΔS, ΔU, ΔV를 예측

    전체 흐름:
      1. ocean 입력과 atmos 입력을 채널 방향으로 결합
      2. ConvLSTM backbone으로 시계열 인코딩
      3. 마지막 시점 hidden state 추출
      4. CNN head로 미래 변화량 출력
    """

    def __init__(
        self,
        ocean_channels: int = 4,
        atmos_channels: int = 5,
        hidden_channels: tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        output_channels: int = 4,
        lead_steps: int = 3,
    ):
        super().__init__() # 몇 개의 미래 시점, 변수를 예측할지 저장
        self.lead_steps = lead_steps
        self.output_channels = output_channels

        input_channels = ocean_channels + atmos_channels # ocean 4 + atmos 5 = 9채널 입력
        self.backbone = ConvLSTM( # 시계열 정보 읽음
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        last_hidden = hidden_channels[-1]
        # CNN head:
        # 마지막 hidden state를 받아 각 lead time의 ΔT, ΔS, ΔU, ΔV를 출력한다.
        self.head = nn.Sequential(
            nn.Conv2d(last_hidden, last_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_hidden, lead_steps * output_channels, kernel_size=1),
        ) # 1x1 conv: 공간 해상도는 유지하고, 필요한 출력 채널 수만 맞춤

# ConvLSTM backbone 뒤에 얕은 CNN head를 두어, 시계열 특징을 실제 예측 변수 공간으로 사상하도록 구성


    def forward(self, x_ocean: torch.Tensor, x_atmos: torch.Tensor) -> torch.Tensor:
        if x_ocean.ndim != 5 or x_atmos.ndim != 5: # 입력 5차원인지 확인
            raise ValueError("x_ocean and x_atmos must be [B, T, C, H, W]")
        if x_ocean.shape[:2] != x_atmos.shape[:2]:
            raise ValueError("ocean and atmos must share batch/time dimensions")
        if x_ocean.shape[3:] != x_atmos.shape[3:]: # 모델 안정성을 위한 안전장치
            raise ValueError("ocean and atmos must share lat/lon dimensions")

        # [순서 1] 해양 입력과 대기 입력을 채널 방향으로 합친다.
        # 예: 4채널 ocean + 5채널 atmos -> 9채널 입력
        x = torch.cat([x_ocean, x_atmos], dim=2)

        # [순서 2] ConvLSTM이 시계열 전체를 읽고
        # 각 시점별 hidden feature를 만든다.
        sequence = self.backbone(x)

        # [순서 3] 마지막 시점 hidden state는
        # "과거 t-2:t 정보를 요약한 표현"으로 사용한다.
        last_hidden = sequence[:, -1]

        # [순서 4] 마지막 hidden state에서 미래 변화량을 뽑는다.
        logits = self.head(last_hidden)
        batch_size, _, height, width = logits.shape

        # [순서 5] 출력 채널을
        # [lead_steps, output_channels] 형태로 다시 정리한다.
        y_hat = logits.view(
            batch_size,
            self.lead_steps,
            self.output_channels,
            height,
            width,
        )
        return y_hat

# 출력은 lead_steps=3, output_channels=4 이면 [B, 3, 4, H, W]가 된다.

def main() -> None: # 기능 점검
    """
    실제 데이터가 없어도 구조 확인용으로 바로 실행 가능.

    이 함수는 랜덤 텐서를 만들어
    "모델이 어떤 입력을 받아 어떤 shape의 출력을 내는지"만 확인한다.
    """
    # 사용 가능한 가속 장치를 우선순위대로 선택한다.
    # 1) NVIDIA GPU가 있으면 CUDA
    # 2) Mac GPU(Metal)가 가능하면 MPS
    # 3) 둘 다 없으면 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    batch_size = 2
    time_steps = 3
    ocean_channels = 4
    atmos_channels = 5
    height = 32
    width = 64
    lead_steps = 3 # shape 예시

    model = OceanConvLSTMForecast( # 메인 모델 객체 만들기
        ocean_channels=ocean_channels,
        atmos_channels=atmos_channels,
        hidden_channels=(32, 32),
        output_channels=4,
        lead_steps=lead_steps,
    ).to(device) # GPU/CPU에 올림

    # 예시 입력:
    # batch=2, time=3, ocean channel=4, atmos channel=5
    x_ocean = torch.randn( # shape만 맞으면 구조 검증 가능
        batch_size,
        time_steps,
        ocean_channels,
        height,
        width,
        device=device,
    )
    x_atmos = torch.randn(
        batch_size,
        time_steps,
        atmos_channels,
        height,
        width,
        device=device,
    )

    with torch.no_grad():
        y_hat = model(x_ocean, x_atmos) # 최종 예측 shape 생성

    # 기대 출력:
    # [batch, lead_steps, output_channels, lat, lon]
    print("device:", device) # 어떤 device인지
    print(model)
    print("x_ocean shape:", tuple(x_ocean.shape))
    print("x_atmos shape:", tuple(x_atmos.shape)) # 입력 shape
    print("y_hat shape:", tuple(y_hat.shape)) # 출력 shape


if __name__ == "__main__":
    main()
