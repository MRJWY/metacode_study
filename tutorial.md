# LLM Inference Engine Study — Inside SGLang

> **LLM을 사용하는 법이 아니라, LLM이 어떻게 빠르게 서빙되는지를 이해한다.**

**기간:** 4주  
**구성:** 주 2회 × 총 8 Sessions  
**권장 시간:** Session당 약 2시간  
**형태:** 개념 학습 + 코드 리딩 + GPU 실험 + Mini Project  
**주요 프레임워크:** SGLang  
**GPU 환경:** RunPod — 학생당 약 **$15 Credit**

---

## 1. 스터디 소개

LLM 애플리케이션을 만드는 것은 점점 쉬워지고 있지만, 하나의 질문이 여전히 남습니다.

> **“모델은 실제 서비스 환경에서 어떻게 수백 개의 요청을 동시에 처리할까?”**

단순한 `model.generate()`에서 production inference server로 넘어가면 새로운 문제가 등장합니다.

- 왜 첫 번째 토큰은 늦고 이후 토큰은 빠를까?
- 왜 decode 단계에서는 GPU 연산량보다 메모리 대역폭이 중요할까?
- KV Cache는 왜 필요하고, 어떻게 GPU 메모리를 차지할까?
- 여러 요청은 어떻게 하나의 batch로 합쳐질까?
- 긴 prompt 하나가 다른 사용자의 요청을 느리게 만들 수 있을까?
- 동일한 system prompt를 가진 요청의 계산을 재사용할 수 있을까?
- throughput을 높이면 latency는 항상 나빠질까?
- “이 최적화가 빨라졌다”는 것을 어떻게 제대로 증명할까?

이 스터디에서는 이러한 질문을 **SGLang을 직접 실행하고 측정하면서** 탐구합니다.

SGLang 자체의 사용법을 외우는 것이 목적은 아닙니다.

SGLang을 하나의 실제 사례로 삼아,

**LLM inference engine을 구성하는 보편적인 아이디어와 trade-off를 이해하는 것**

이 최종 목표입니다.

---

## 2. Learning Goals

4주가 끝났을 때 다음 질문에 자신의 말로 답할 수 있는 것을 목표로 합니다.

### Inference Fundamentals

- Prefill과 Decode는 무엇이 다른가?
- TTFT, TPOT/ITL, throughput은 각각 무엇을 의미하는가?
- LLM inference는 언제 compute-bound이고 언제 memory-bound인가?

### Memory & KV Cache

- KV Cache가 왜 필요한가?
- Context length와 concurrent requests가 증가하면 메모리는 어떻게 증가하는가?
- Prefix caching은 어떤 workload에서 효과적인가?
- SGLang의 RadixAttention은 무엇을 해결하려는가?

### Scheduling & Batching

- Continuous batching은 static batching과 무엇이 다른가?
- Scheduler는 waiting/running requests를 어떻게 관리해야 하는가?
- 긴 prompt가 decode latency에 영향을 미치는 이유는 무엇인가?
- Throughput과 tail latency 사이에는 어떤 trade-off가 있는가?

### Optimization

- Attention kernel, CUDA Graph, quantization 등의 최적화는 어느 계층에서 동작하는가?
- 최적화를 켰는데 오히려 느려질 수 있는 이유는 무엇인가?
- Workload에 따라 최적의 configuration이 달라지는 이유는 무엇인가?

### Benchmarking

- 좋은 inference benchmark를 어떻게 설계하는가?
- 평균 latency만 측정하면 왜 부족한가?
- p50 / p95 / p99, TTFT, ITL, throughput을 어떻게 해석하는가?
- 성능 향상과 GPU 비용을 함께 어떻게 비교하는가?

---

## 3. 스터디의 원칙

### 3.1 “기능”보다 “왜”를 공부한다

예를 들어,

```text
--disable-radix-cache
```

라는 옵션 자체를 기억하는 것은 중요하지 않습니다.

대신 다음을 설명할 수 있어야 합니다.

> Radix Cache가 없으면 어떤 계산이 반복되고,  
> shared prefix가 많을수록 왜 caching의 가치가 커지는가?

### 3.2 모든 주장에는 측정값이 있어야 한다

스터디 중 다음 문장은 지양합니다.

> “이 옵션을 켜니까 빨라진 것 같다.”

대신 다음처럼 이야기합니다.

> 동일한 workload에서 cache를 활성화했을 때  
> p50 TTFT가 A ms → B ms로 감소했다.

**System optimization = hypothesis + experiment + measurement**

입니다.

### 3.3 한 번에 하나의 변수만 바꾼다

가능하면 다음 조건은 고정합니다.

- GPU
- model
- input length
- output length
- request count
- concurrency
- sampling configuration

그리고 **한 가지 변수만 변경**합니다.

그래야 무엇이 성능 변화를 만들었는지 설명할 수 있습니다.

### 3.4 GPU는 실험할 때만 사용한다

RunPod의 GPU는 개발 환경이 아니라 **측정 장비**라고 생각합니다.

코드 작성, 결과 분석, 그래프 생성, README 작성은 가능하면 로컬에서 진행합니다.

GPU에서는:

1. 환경 준비
2. benchmark 실행
3. 결과 저장
4. 종료

만 수행합니다.

---

## 4. Standard Experimental Setup

학생마다 서로 다른 실험 환경을 사용하면 결과 비교가 어려워지므로 가능한 한 공통 환경을 사용합니다.

### Default

- Linux
- NVIDIA GPU
- Python
- PyTorch
- Hugging Face Transformers
- SGLang
- Git

### Model

기본 실습에서는 **0.5B–4B 정도의 작은 모델**을 우선 사용합니다.

큰 모델의 품질을 평가하는 스터디가 아니기 때문에 굳이 7B/14B/70B 모델을 사용할 필요가 없습니다.

작은 모델을 사용하면:

- model loading 비용 감소
- benchmark 반복 횟수 증가
- OOM 위험 감소
- GPU 비용 감소

라는 장점이 있습니다.

---

## 5. 우리가 사용할 핵심 Metrics

전체 스터디에서 가능한 한 동일한 metric을 사용합니다.

### Latency

**TTFT — Time To First Token**

Request를 보낸 시점부터 첫 token을 받을 때까지 걸리는 시간.

Prefill 성능과 scheduling의 영향을 크게 받습니다.

**ITL — Inter-Token Latency**

출력 token과 다음 token 사이의 시간.

사용자가 streaming generation을 경험할 때 체감하는 생성 속도와 관련이 있습니다.

**TPOT — Time Per Output Token**

첫 token 이후 output token 한 개를 처리하는 평균 시간.

**E2E Latency**

Request부터 전체 generation 완료까지의 시간.

### Throughput

- requests / sec
- output tokens / sec
- total tokens / sec

### Distribution

Latency는 평균만 보지 않습니다.

가능하면 함께 기록합니다.

- mean
- p50
- p95
- p99

---

## 6. Curriculum Overview

| Week | Session | Topic | 핵심 질문 |
|---|---|---|---|
| Week 1 | 1 | LLM Inference Fundamentals | 하나의 token은 어떻게 만들어지는가? |
| Week 1 | 2 | Inside an Inference Engine | `generate()`와 inference server는 무엇이 다른가? |
| Week 2 | 3 | KV Cache & RadixAttention | 이전 계산을 어떻게 재사용할 수 있는가? |
| Week 2 | 4 | Continuous Batching & Scheduling | 여러 요청을 GPU에 어떻게 넣을 것인가? |
| Week 3 | 5 | Execution & Optimization | 실제 GPU에서 무엇을 최적화하는가? |
| Week 3 | 6 | Benchmarking & Profiling | “빨라졌다”를 어떻게 증명하는가? |
| Week 4 | 7 | Mini Project Sprint | 하나의 inference 질문을 실험으로 검증한다 |
| Week 4 | 8 | Demo Day & System Review | 결과를 설명하고 전체 시스템을 연결한다 |

---

# Week 1 — From `generate()` to an Inference Engine

## Session 1. LLM Inference Fundamentals

### 핵심 질문

> **LLM은 하나의 token을 생성할 때 실제로 무슨 일을 하는가?**

### Concepts

- Autoregressive generation
- Prompt processing
- Prefill vs Decode
- Forward pass
- Logits → Sampling → Next Token
- KV Cache의 필요성
- Compute-bound vs Memory-bound
- Latency vs Throughput

### Hands-on

Hugging Face Transformers를 이용해 inference engine이 없는 가장 단순한 baseline을 만듭니다.

실험:

```text
Experiment A
batch = 1

Experiment B
batch = 4

Experiment C
batch = 8
```

각 실험에서 기록합니다.

```text
input tokens
output tokens
total latency
tokens/sec
GPU memory
```

가능하면 generation을 직접 반복하면서,

```text
forward
→ logits
→ sample
→ append token
→ forward
```

루프가 어떻게 동작하는지도 확인합니다.

### 생각해볼 것

- Batch를 키우면 항상 한 request의 latency가 좋아질까?
- 왜 throughput은 증가하는데 latency는 증가할 수 있을까?
- Prompt가 길어질 때와 output이 길어질 때 병목은 동일할까?

### Deliverable

`session01-baseline/`

- `baseline.py`
- benchmark 결과
- 간단한 그래프 1개
- 5줄 이내의 observation

---

## Session 2. Inside an Inference Engine — SGLang

### 핵심 질문

> **왜 `model.generate()`만으로 production serving을 만들지 않을까?**

### Concepts

Production inference server에는 모델 외에도 여러 구성 요소가 필요합니다.

```text
Client
  │
  ▼
Tokenizer
  │
  ▼
Request Queue
  │
  ▼
Scheduler
  │
  ▼
Batch
  │
  ▼
Model Runner
  │
  ▼
KV Cache / Memory Pool
  │
  ▼
GPU
```

여기서 특히 살펴볼 부분은 다음과 같습니다.

- Request lifecycle
- Tokenization
- Scheduler
- Continuous batching
- Model execution
- KV Cache management
- Streaming response

### Hands-on 1 — First SGLang Server

SGLang server를 직접 실행합니다.

그리고 Python 또는 OpenAI-compatible client를 이용하여 request를 전송합니다.

확인할 것:

```text
single request
streaming request
multiple requests
concurrent requests
```

### Hands-on 2 — Baseline Comparison

Session 1의 Transformers baseline과 동일하거나 최대한 유사한 workload를 SGLang으로 실행합니다.

비교:

| Engine | Concurrency | TTFT | Throughput | Memory |
|---|---:|---:|---:|---:|
| Transformers | 1 | | | |
| SGLang | 1 | | | |
| SGLang | 8 | | | |

단순히 “누가 빠른가”보다 **왜 차이가 발생하는지**를 이야기합니다.

### Code Reading

Full SGLang 전체 코드를 읽는 대신 `mini-sglang`과 같은 작은 구현을 이용해 시스템의 큰 흐름을 먼저 파악합니다.

### Deliverable

- SGLang server 실행
- Transformers vs SGLang baseline
- **“Request 하나가 들어와 token 하나가 나오기까지”** architecture diagram

---

# Week 2 — Memory and Scheduling

## Session 3. KV Cache & RadixAttention

### 핵심 질문

> **이미 계산한 prompt를 다시 계산하지 않을 수 있을까?**

### Concepts

Transformer decode 과정에서 이전 token의 Key/Value를 다시 계산하지 않기 위해 KV Cache를 사용합니다.

살펴볼 내용:

- KV Cache란?
- KV Cache가 없을 경우
- Context length와 memory
- Requests와 KV Cache
- Memory fragmentation
- Paged KV Cache
- Prefix caching
- Cache eviction
- Radix tree
- RadixAttention

### Example Workload

다음 요청들이 있다고 생각해봅니다.

```text
System Prompt: You are a helpful programming assistant...
Question A

System Prompt: You are a helpful programming assistant...
Question B

System Prompt: You are a helpful programming assistant...
Question C
```

세 request는 긴 prefix를 공유합니다.

질문:

> System Prompt를 매번 처음부터 계산해야 할까?

### Hands-on — Prefix Cache Experiment

두 종류의 workload를 만듭니다.

#### Workload A

서로 완전히 다른 prompt

```text
A
B
C
D
...
```

#### Workload B

긴 shared prefix + 짧은 서로 다른 question

```text
PREFIX + A
PREFIX + B
PREFIX + C
PREFIX + D
...
```

가능하면 각각에서:

```text
Radix Cache ON
Radix Cache OFF
```

를 비교합니다.

### Measure

- TTFT
- throughput
- cache warm / cold 차이
- prefix length에 따른 변화

### Deliverable

그래프 하나를 만듭니다.

```text
Shared Prefix Length
        ↓

TTFT / Throughput
```

그리고 다음 질문에 답합니다.

> **“어떤 workload에서 prefix caching이 가장 유용한가?”**

---

## Session 4. Continuous Batching & Scheduling

### 핵심 질문

> **길이가 모두 다른 request를 GPU에서 어떻게 동시에 처리할까?**

### Static Batch의 문제

다음 네 request가 동시에 들어왔다고 생각해봅니다.

```text
A → 20 tokens
B → 200 tokens
C → 50 tokens
D → 500 tokens
```

Traditional static batching에서는 완료 시점이 서로 다른 sequence를 효율적으로 처리하기 어렵습니다.

Inference engine은 request가 들어오고 끝나는 동안 batch를 계속 재구성합니다.

이를 중심으로 **continuous batching**을 공부합니다.

### Concepts

- Static batching
- Continuous batching
- Request queue
- Running requests
- Waiting requests
- Scheduling
- Token budget
- Head-of-line blocking
- Throughput vs latency
- Tail latency
- Chunked Prefill

### 중요한 상황

```text
User A
Prompt = 10,000 tokens

User B
Prompt = 30 tokens
```

User A의 긴 prefill이 GPU를 오랫동안 점유한다면 User B의 decode가 늦어질 수 있습니다.

이 문제를 해결하기 위한 아이디어 중 하나가 **Chunked Prefill**입니다.

### Hands-on — Load Test

동일한 server에 concurrency를 바꾸면서 request를 전송합니다.

```text
concurrency = 1
concurrency = 2
concurrency = 4
concurrency = 8
concurrency = 16
```

각각 기록합니다.

```text
request throughput
token throughput
TTFT p50
TTFT p95
ITL
```

### 두 번째 실험

Mixed workload를 만듭니다.

```text
short prompt + short output
long prompt  + short output
short prompt + long output
```

질문:

> 어떤 workload가 다른 request에 가장 큰 영향을 주는가?

### Deliverable

두 개의 그래프를 만듭니다.

```text
Concurrency → Throughput

Concurrency → p95 TTFT
```

그리고 **sweet spot이라고 생각하는 concurrency**를 하나 선택하고 근거를 설명합니다.

---

# Week 3 — Optimization and Measurement

## Session 5. Execution & Optimization

### 핵심 질문

> **Inference engine은 GPU에서 정확히 무엇을 최적화하고 있는가?**

### Optimization Stack

Inference performance는 한 가지 기술로 결정되지 않습니다.

```text
Application
    │
API / Streaming
    │
Scheduler
    │
Batching
    │
KV Cache
    │
Model Execution
    │
Attention / GEMM Kernels
    │
CUDA
    │
GPU
```

각 계층에서 서로 다른 optimization이 동작합니다.

### Concepts

- Attention kernel
- FlashAttention / FlashInfer 계열 backend
- CUDA Graph
- Kernel launch overhead
- Memory bandwidth
- Weight quantization
- KV Cache precision
- Chunked prefill
- Speculative decoding

### Speculative Decoding

일반적인 autoregressive decoding은:

```text
1 forward
→ 1 token
```

을 반복합니다.

Speculative decoding에서는 작은 draft mechanism이 여러 후보 token을 제안하고 target model이 이를 검증합니다.

핵심 질문은 다음입니다.

> Forward 횟수를 줄여 얻는 이득이  
> draft와 verification에 추가되는 비용보다 클까?

단, **본 스터디에서는 speculative decoding을 필수 실습으로 요구하지 않습니다.**

모델 compatibility와 추가 VRAM 때문에 학생당 $15 예산에서는 비용이 커질 수 있기 때문입니다.

### Hands-on — Ablation Study

각자 **하나의 optimization만 선택**합니다.

예:

```text
CUDA Graph ON/OFF
```

또는

```text
Attention backend A/B
```

또는

```text
BF16 vs quantized model
```

또는

```text
Cache configuration A/B
```

### Rule

```text
Before
vs
After
```

를 비교할 때 나머지 조건을 동일하게 유지합니다.

### Deliverable

**1-page Optimization Note**

```text
Hypothesis

Experimental Setup

Baseline

Optimization

Result

Why?
```

---

## Session 6. Benchmarking & Profiling

### 핵심 질문

> **“30% 빨라졌다”라는 말을 어떻게 믿을 수 있을까?**

### Part 1 — Benchmark Design

좋은 benchmark는 다음 조건을 만족해야 합니다.

#### Reproducible

누가 다시 실행해도 비슷한 결과가 나와야 합니다.

#### Controlled

비교하려는 변수 외의 조건이 동일해야 합니다.

#### Representative

실제 우리가 관심 있는 workload와 유사해야 합니다.

### 흔한 Benchmark 실수

```text
❌ cold start와 warm server 비교

❌ input 길이가 서로 다름

❌ output 길이가 서로 다름

❌ model이 다름

❌ GPU가 다름

❌ concurrency가 다름

❌ 평균 latency만 비교

❌ 한 번만 실행
```

### Standard Benchmark Matrix

다음과 같은 작은 matrix를 사용합니다.

```text
Input Length
128 / 512 / 2048

Output Length
64 / 256

Concurrency
1 / 4 / 16
```

모든 조합을 반드시 실행할 필요는 없습니다.

**자신의 hypothesis를 검증하는 최소한의 실험**을 선택합니다.

### Part 2 — Profiling

성능이 느리다는 사실만 아는 것과 어디가 느린지 아는 것은 다릅니다.

관찰할 수 있는 항목:

- CPU scheduling
- tokenization
- queueing
- prefill
- decode
- GPU utilization
- memory usage
- network/API overhead

### Part 3 — Cost Performance

Cloud 환경에서는 가장 빠른 configuration이 반드시 가장 좋은 것은 아닙니다.

예를 들어 다음 값을 계산해봅니다.

```text
tokens / dollar
```

또는

```text
1M output tokens 당 예상 GPU 비용
```

### Deliverable

`benchmark_report.md`

포함할 것:

- Experiment question
- Environment
- Workload
- Metrics
- Result table
- Graph
- Interpretation
- Limitation

이 문서는 Session 7 Mini Project의 template이 됩니다.

---

# Week 4 — Apply What You Learned

## Session 7. Mini Project Sprint

### 목표

거대한 inference service를 만드는 것이 아닙니다.

**작고 명확한 시스템 질문 하나를 선택하고, 실험으로 답합니다.**

### 좋은 Project Question

> Shared prefix가 길어질수록 Radix Cache의 TTFT 개선 효과는 어떻게 변하는가?

좋습니다.

### 너무 큰 Project Question

> SGLang 기반 production chatbot을 구축한다.

4주 프로젝트로는 범위가 너무 큽니다.

### Project Examples

#### A. Radix Cache Explorer

질문:

> Shared prefix 비율이 어느 정도부터 caching이 유리해지는가?

변수:

```text
prefix length
cache on/off
request count
```

#### B. Continuous Batching Explorer

질문:

> Concurrency 증가가 throughput과 tail latency에 미치는 영향은?

변수:

```text
concurrency
input length
output length
```

#### C. Long Prompt Interference

질문:

> 긴 prefill request가 짧은 request의 decode latency에 얼마나 영향을 미치는가?

비교:

```text
short-only workload

vs

short + long mixed workload
```

#### D. Optimization Ablation

하나의 optimization을 선택합니다.

예:

```text
CUDA Graph
Attention backend
Quantization
Prefix cache
Chunked prefill
```

그리고 baseline과 비교합니다.

#### E. Cost-Aware Configuration Finder

질문:

> $1 안에서 가장 많은 token을 생성하는 configuration은 무엇인가?

출력:

```text
configuration
throughput
latency
tokens/$
```

#### F. mini-SGLang Hack

조금 더 시스템 구현에 관심 있는 학생용입니다.

예:

- scheduler 코드 수정
- cache policy 변경
- scheduling rule 변경
- instrumentation 추가

변경 전후를 benchmark합니다.

#### G. SGLang vs Another Engine

동일한:

```text
GPU
Model
Input
Output
Concurrency
```

에서 SGLang과 다른 inference implementation을 비교합니다.

단순 순위가 아니라,

> **왜 workload에 따라 차이가 발생하는가?**

를 설명하는 것이 핵심입니다.

### Project Required Structure

모든 프로젝트는 다음 형식을 따릅니다.

#### 1. Question

무엇을 알고 싶은가?

#### 2. Hypothesis

결과가 어떻게 나올 것으로 예상하는가?

#### 3. Baseline

비교 기준은 무엇인가?

#### 4. Change

무엇을 변경했는가?

#### 5. Measurement

어떤 metric으로 평가했는가?

#### 6. Result

실제 결과는 무엇인가?

#### 7. Explanation

왜 이런 결과가 나왔다고 생각하는가?

#### 8. Limitation

이 실험으로 말할 수 없는 것은 무엇인가?

### Session 7 Deliverable

Project repository의 최소 형태:

```text
project/
├── README.md
├── run.sh
├── benchmark.py
├── configs/
├── results/
│   └── results.csv
└── figures/
    └── result.png
```

---

## Session 8. Demo Day & System Review

마지막 Session은 새로운 내용을 배우는 시간이 아닙니다.

4주 동안 공부한 내용을 하나의 시스템으로 연결합니다.

### Part 1 — Project Presentation

학생당 권장 발표 시간:

```text
7 min presentation
+
3 min Q&A
```

발표 구조:

#### Problem

무슨 질문을 던졌는가?

#### Hypothesis

어떤 결과를 예상했는가?

#### Experiment

어떻게 검증했는가?

#### Result

무엇을 발견했는가?

#### Explanation

왜 그런 결과가 나왔다고 생각하는가?

#### What I would test next

시간과 GPU가 더 있다면 무엇을 실험할 것인가?

### Part 2 — Final System Walkthrough

마지막으로 전체 inference pipeline을 다시 봅니다.

```text
                 ┌──────────────┐
                 │    Client    │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Tokenization │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │Request Queue │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Scheduler   │
                 └──────┬───────┘
                        │
               Continuous Batching
                        │
                        ▼
              ┌───────────────────┐
              │   Model Runner    │
              └────────┬──────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
       Prefill                    Decode
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
                ┌────────────┐
                │  KV Cache  │
                │Radix Cache │
                └─────┬──────┘
                      │
                      ▼
               Attention / GEMM
                      │
                      ▼
                     GPU
```

그리고 각 component가 다음 metric에 어떻게 영향을 줄 수 있는지 이야기합니다.

```text
TTFT
ITL
Throughput
Memory
p99 Latency
Cost
```

---

## 7. RunPod Budget

각 학생에게 주어진 GPU budget은 약:

# **$15**

입니다.

하지만 $15를 모두 사용하는 것이 목표가 아닙니다.

**적은 비용으로 좋은 실험을 설계하는 것도 inference engineering의 일부입니다.**

### 권장 Budget Allocation

실제 GPU 가격에 따라 달라질 수 있으므로 아래 금액은 **결제 예상액이 아니라 스터디 운영 상한선**입니다.

| Session | 권장 Budget |
|---|---:|
| Session 1 | $0.8 |
| Session 2 | $1.0 |
| Session 3 | $1.0 |
| Session 4 | $1.2 |
| Session 5 | $1.2 |
| Session 6 | $1.2 |
| Session 7 | $1.8 |
| Final Project 추가 실험 | $3.0 |
| **예비 Budget** | **$3.8** |
| **Total** | **$15.0** |

### GPU Budget Rules

#### Rule 1

GPU를 띄운 상태에서 코드를 처음부터 작성하지 않습니다.

먼저 local 환경에서 script를 준비합니다.

#### Rule 2

실험 조건을 미리 결정합니다.

잘못된 예:

```text
GPU 켜기
→ 뭘 할지 생각하기
→ 코드 작성
→ 디버깅
```

좋은 예:

```text
Experiment 설계
→ 코드 준비
→ GPU 켜기
→ 실행
→ 결과 저장
→ GPU 종료
```

#### Rule 3

큰 모델은 필요할 때만 사용합니다.

이 스터디에서 중요한 것은

```text
Model Quality ❌

Inference Behavior ✅
```

입니다.

#### Rule 4

모든 학생은 자신의 GPU 사용 기록을 남깁니다.

예:

```text
Date
GPU
Experiment
Start
End
Approx. Cost
```

---

## 8. Before Week 1

Session에 포함되지 않는 사전 Setup입니다.

다음 명령을 사용할 수 있어야 합니다.

```bash
git clone
cd
ls
python
pip
nvidia-smi
```

그리고 기본적인 Python script를 실행할 수 있어야 합니다.

### 권장 사전 지식

#### Required

- Python 기본 문법
- Linux command line 기본
- Transformer의 큰 구조
- Autoregressive generation 개념

#### Helpful but Not Required

- PyTorch
- CUDA
- Hugging Face Transformers
- REST API
- asyncio
- profiling
- Docker

CUDA kernel을 직접 작성해본 경험은 필요하지 않습니다.

---

## 9. Repository Convention

전체 스터디 repository는 가능한 한 다음 구조를 유지합니다.

```text
inference-study/
│
├── session01/
├── session02/
├── session03/
├── session04/
├── session05/
├── session06/
│
├── project/
│
├── results/
│
└── README.md
```

각 Session에는 최소한 다음 파일을 남깁니다.

```text
README.md

run.sh 또는 실행 방법

results.csv/json

간단한 observation
```

---

## 10. How We Discuss Results

스터디에서 결과를 공유할 때 다음 네 문장으로 설명하는 습관을 만듭니다.

> **I expected ...**

> **I changed ...**

> **I measured ...**

> **I found ...**

성능 숫자 자체보다 이 네 문장을 명확하게 말할 수 있는 것이 중요합니다.

---

## 11. Completion Criteria

스터디가 끝난 뒤 다음 항목에 대부분 답할 수 있다면 성공입니다.

- [ ] Prefill과 Decode의 차이를 설명할 수 있다.
- [ ] TTFT와 ITL/TPOT의 의미를 설명할 수 있다.
- [ ] KV Cache가 필요한 이유를 설명할 수 있다.
- [ ] Context length가 memory에 미치는 영향을 설명할 수 있다.
- [ ] Prefix caching과 RadixAttention의 목적을 설명할 수 있다.
- [ ] Continuous batching이 필요한 이유를 설명할 수 있다.
- [ ] Scheduler가 해결해야 하는 문제를 설명할 수 있다.
- [ ] Throughput과 latency 사이의 trade-off를 설명할 수 있다.
- [ ] SGLang inference server를 직접 실행할 수 있다.
- [ ] Concurrent workload를 만들어 benchmark할 수 있다.
- [ ] p50 / p95 / p99 latency를 측정하고 해석할 수 있다.
- [ ] 하나의 optimization을 controlled experiment로 평가할 수 있다.
- [ ] 성능과 GPU 비용을 함께 비교할 수 있다.
- [ ] 자신이 측정한 결과의 limitation을 설명할 수 있다.
- [ ] 작은 inference systems project를 재현 가능한 형태로 공개할 수 있다.

---

## 12. Final Takeaway

이 스터디가 끝났을 때 목표는

> **“SGLang을 써봤다.”**

가 아닙니다.

목표는 새로운 inference framework를 보더라도 다음 질문을 던질 수 있는 것입니다.

> Batch는 어떻게 만드는가?

> Request는 어떻게 schedule하는가?

> KV Cache는 어떻게 관리하는가?

> 이전 계산은 어떻게 재사용하는가?

> GPU에서는 무엇이 병목인가?

> Throughput과 latency 사이에서 어떤 선택을 하는가?

> 그리고 그 선택이 정말 더 좋은지는 어떻게 측정하는가?

SGLang은 이 질문들을 실제 코드와 GPU 위에서 공부하기 위한 도구입니다.

# **Build less. Measure more. Understand why.**