# Week 1 강의 노트 — From `generate()` to an Inference Engine

> **이번 주의 한 문장:** LLM이 token 하나를 만드는 과정을 끝까지 따라가 보고, 왜 그 과정만으로는 production serving이 안 되는지를 직접 측정으로 확인한다.

| | Session 1 | Session 2 |
|---|---|---|
| 주제 | LLM Inference Fundamentals | Inside an Inference Engine — SGLang |
| 핵심 질문 | LLM은 token 하나를 만들 때 실제로 무슨 일을 하는가? | 왜 `model.generate()`만으로 production serving을 만들지 않는가? |
| 권장 시간 | 약 2시간 | 약 2시간 |
| GPU Budget | $0.8 | $1.0 |
| Deliverable | `session01-baseline/` | `session02-sglang/` |

---

## Week 1 학습 목표

이번 주가 끝나면 다음을 **자신의 말로** 설명할 수 있어야 합니다.

- [ ] Autoregressive generation loop를 한 단계씩 설명할 수 있다.
- [ ] Prefill과 Decode의 차이를 연산량·메모리 관점에서 설명할 수 있다.
- [ ] KV Cache가 없으면 무엇이 반복 계산되는지 설명할 수 있다.
- [ ] KV Cache 크기를 model config로부터 직접 계산할 수 있다.
- [ ] Decode가 왜 memory-bound인지, batch가 왜 그 상황을 바꾸는지 설명할 수 있다.
- [ ] TTFT / ITL / TPOT / E2E latency를 정의하고 어디서 측정하는지 말할 수 있다.
- [ ] Inference server의 구성 요소와 request lifecycle을 그림으로 그릴 수 있다.
- [ ] SGLang server를 직접 띄우고 request를 보낼 수 있다.

---

## 사전 준비 (Session 시작 전 로컬에서)

```bash
git clone <study-repo>
cd inference-study
python -V          # 3.10+ 권장
pip -V
```

GPU에 올라가기 **전에** 다음이 준비되어 있어야 합니다. (스터디 원칙 3.4 — GPU는 측정 장비)

- `session01/baseline.py` 초안
- 실험 조건 표 (아래 §1.6에서 확정)
- 결과를 저장할 `results.csv` 스키마

---

---

# Session 1. LLM Inference Fundamentals

## 진행 순서 (권장 2시간)

| 시간 | 내용 |
|---|---|
| 0:00–0:15 | 오프닝: 오늘의 질문, Week 1 전체 지도 |
| 0:15–0:50 | 개념 강의 §1.1–§1.4 |
| 0:50–1:05 | 휴식 / 로컬 코드 최종 점검 |
| 1:05–1:40 | Hands-on (GPU ON → 실험 A/B/C → GPU OFF) |
| 1:40–2:00 | 결과 공유, 토론 질문 |

---

## §1.1 하나의 token은 어떻게 만들어지는가

LLM generation은 결국 다음 루프의 반복입니다.

```text
tokens = tokenize(prompt)

loop:
    logits = model.forward(tokens)      # [batch, seq_len, vocab_size]
    next_logits = logits[:, -1, :]      # 마지막 위치만 사용
    next_token = sample(next_logits)    # greedy / temperature / top-p
    tokens.append(next_token)
    if next_token == EOS: break
```

여기서 반드시 짚고 갈 점 두 가지입니다.

**(1) 모델은 매 step마다 vocabulary 전체에 대한 확률 분포를 만든다.**
Vocab이 15만이면 step마다 15만 개의 logit이 나오고, 우리는 그중 **딱 한 개**만 씁니다. 나머지는 버려집니다.

**(2) 모델은 "다음 token 하나"만 예측한다.**
"문장을 생성한다"는 것은 이 예측을 수십~수천 번 반복한 결과일 뿐입니다. 그래서 output token이 512개면 forward pass도 (원칙적으로) 512번입니다.

> 💡 **여기서 첫 번째 성능 직관이 나옵니다.**
> Output length는 latency에 거의 선형으로 영향을 줍니다. Prompt length는 다른 방식으로 영향을 줍니다. 이 둘의 병목은 서로 다릅니다.

### Sampling에 대한 스터디 규칙

Sampling 전략(temperature, top-p, top-k)은 이 스터디의 관심사가 **아닙니다**. 하지만 성능 측정에는 영향을 줍니다. 무작위성이 있으면 output length가 실행마다 달라져 비교가 불가능해집니다.

따라서 **모든 benchmark에서 다음을 고정합니다.**

```text
temperature = 0        (greedy)
ignore_eos = True 또는 min_new_tokens = max_new_tokens
```

이렇게 하면 모든 request가 정확히 같은 개수의 token을 생성하므로, latency 차이가 "생성 길이가 달라서"가 아니라 "시스템이 달라서" 생긴 것임을 보장할 수 있습니다.

---

## §1.2 Prefill vs Decode

Generation은 성격이 완전히 다른 두 단계로 나뉩니다.

```text
Prompt: "The capital of France is"   (5 tokens)

┌─────────────── PREFILL ───────────────┐
5개 token을 한 번의 forward로 처리
→ 각 위치의 K, V를 계산해서 cache에 저장
→ 마지막 위치의 logit으로 첫 token 생성
                                      │
                                      ▼
┌─────────────── DECODE ────────────────┐
token 1개를 입력 → forward → token 1개 출력
token 1개를 입력 → forward → token 1개 출력
...  (output length만큼 반복)
```

| | Prefill | Decode |
|---|---|---|
| 한 번에 처리하는 token 수 | prompt 길이 (수백~수만) | 1 (request당) |
| Forward pass 횟수 | 1회 | output length만큼 |
| GPU 연산 형태 | 큰 행렬 × 행렬 (GEMM) | 행렬 × 벡터 (GEMV에 가까움) |
| 병목 | **연산량 (compute-bound)** | **메모리 대역폭 (memory-bound)** |
| 대응 metric | **TTFT** | **ITL / TPOT** |
| GPU 활용률 | 높음 | 낮음 (batch가 작을 때) |

> **"왜 첫 번째 token은 늦고 이후 토큰은 빠른가?"** (스터디 소개의 첫 질문)
>
> 첫 token은 prompt 전체를 처리하는 prefill을 기다려야 합니다. 이후 token들은 token 하나씩만 forward하면 되고, 앞선 계산은 KV Cache에 저장되어 있습니다. **단, 이건 "혼자 쓸 때" 이야기입니다.** 서버에 다른 요청이 있으면 이야기가 달라집니다 — 그게 Session 4의 주제입니다.

---

## §1.3 KV Cache — 왜 필요한가

### KV Cache가 없다면

Attention은 현재 token의 Query가 **이전 모든 token의 Key/Value**를 봐야 합니다. Cache가 없으면 매 step마다 전체 sequence를 처음부터 다시 forward해야 합니다.

```text
step 1: forward(t1..t5)        →  5 token 처리
step 2: forward(t1..t6)        →  6 token 처리   ← t1~t5를 또 계산
step 3: forward(t1..t7)        →  7 token 처리   ← 또
...
step N: forward(t1..t(5+N-1))
```

총 처리량이 **O(S²)** 로 증가합니다. 반면 K와 V를 저장해 두면 매 step에서 새 token 1개분의 K, V만 계산하면 되므로 **O(S)** 가 됩니다.

**즉 KV Cache는 "메모리를 써서 연산을 사는" 전형적인 trade-off입니다.** 그리고 이 스터디의 절반은 그 대가로 지불한 메모리를 어떻게 관리하느냐에 관한 것입니다.

### KV Cache 크기 계산

```text
KV bytes/token = 2 (K와 V)
               × num_hidden_layers
               × num_key_value_heads
               × head_dim
               × dtype_bytes
```

**예제: Qwen2.5-0.5B-Instruct (FP16/BF16)**

| 항목 | 값 |
|---|---|
| num_hidden_layers | 24 |
| num_key_value_heads | 2 (GQA) |
| head_dim | 64 |
| dtype_bytes | 2 |

```text
2 × 24 × 2 × 64 × 2 = 12,288 bytes ≈ 12 KiB / token
```

여기서 규모 감각을 잡아봅니다.

| 상황 | KV Cache |
|---|---|
| 1 request × 1K tokens | 약 12 MiB |
| 1 request × 32K tokens | 약 384 MiB |
| 64 requests × 2K tokens | 약 1.5 GiB |
| 256 requests × 4K tokens | 약 12 GiB |

> 📌 **Session 실습 과제:** 자신이 사용할 모델의 `config.json`을 열어 위 수식으로 token당 KV 크기를 직접 계산하고 노트에 적으세요. 그리고 "내 GPU 메모리에서 model weight를 빼고 남은 공간에 몇 token이 들어가는가?"를 계산해 보세요. 이 숫자가 Session 3/4에서 계속 등장합니다.

**GQA(Grouped Query Attention)의 의미도 여기서 보입니다.** `num_key_value_heads`가 attention head 수보다 작기 때문에 KV Cache가 크게 줄어듭니다. Qwen2.5-0.5B는 attention head 14개에 KV head 2개이므로, MHA였다면 KV Cache가 7배 컸을 것입니다.

---

## §1.4 Compute-bound vs Memory-bound

이 스터디에서 가장 중요한 개념 하나를 고르라면 이것입니다.

GPU에서 어떤 연산이 걸리는 시간은 둘 중 **더 느린 쪽**이 결정합니다.

```text
time ≈ max( FLOPs / peak_compute ,  bytes_moved / memory_bandwidth )
```

**Arithmetic Intensity** = FLOPs / bytes. 이 값이 GPU의 (peak_compute / bandwidth) 비율보다 크면 compute-bound, 작으면 memory-bound입니다.

예를 들어 A100 80GB는 대략 BF16 312 TFLOPS, 대역폭 2.0 TB/s이므로 임계 비율이 **약 150 FLOP/byte**입니다.

### Decode의 산술 강도는 대략 batch size와 같다

Decode step 하나에서:

- 읽어야 하는 바이트: **model weight 전체** (N개 파라미터 × 2 bytes)
- 수행하는 연산: 파라미터당 곱셈-덧셈 ≈ 2 × N × B FLOPs (B = batch size)

```text
intensity ≈ (2 × N × B) / (2 × N) = B
```

**B = 1이면 intensity가 1입니다.** 임계값 150에 한참 못 미칩니다. 즉 batch 1 decode에서 GPU는 계산을 거의 하지 않고 **weight를 읽는 데 시간을 다 씁니다.**

> 💡 이것이 **"왜 decode 단계에서는 GPU 연산량보다 메모리 대역폭이 중요한가"** 에 대한 답입니다.
>
> 그리고 동시에 **"왜 batching이 그렇게 강력한가"** 에 대한 답이기도 합니다. Weight는 어차피 한 번 읽습니다. 그 한 번에 요청 8개를 함께 처리하면, 읽기 비용이 8개 요청에 분산됩니다. **Batch를 키워도 decode step 시간이 거의 늘지 않는 구간이 존재하는 이유**가 바로 이것입니다.

### Prefill은 반대다

Prefill은 S개 token을 한 번에 처리하므로 intensity ≈ S입니다. Prompt가 512 token이면 이미 임계값을 넘어 **compute-bound**입니다. 그래서 prefill에서는 batch를 키워도 이득이 크지 않고, 오히려 GPU를 오래 점유합니다. (→ Session 4의 head-of-line blocking 문제로 이어집니다.)

| | 지배 요인 | Batch 확대 효과 |
|---|---|---|
| Prefill | 연산량 | 작음 (이미 GPU 포화) |
| Decode | 메모리 대역폭 | **큼** (weight 읽기 비용 분산) |

---

## §1.5 Latency와 Throughput

### 측정 지점

```text
t0        t1                        t2      t3            t4
│         │                         │       │             │
│ request │ ...prefill...           │ tok1  │ tok2 ...    │ 완료
└─────────┴─────────────────────────┴───────┴─────────────┘

TTFT = t2 - t0        (첫 token 도착까지)
ITL  = t3 - t2, ...   (연속한 token 사이 간격들)
TPOT = (t4 - t2) / (output_tokens - 1)
E2E  = t4 - t0
```

| Metric | 사용자 체감 | 주로 영향받는 요소 |
|---|---|---|
| TTFT | "응답이 시작되기까지의 답답함" | prompt 길이, prefill, **queue 대기 시간** |
| ITL / TPOT | "글자가 흘러나오는 속도" | decode, batch 크기, 메모리 대역폭 |
| E2E | 전체 완료 시간 | 위 둘 + output 길이 |
| Throughput | 서비스 운영 비용 | batching, scheduling, 전체 GPU 활용률 |

### 왜 평균만 보면 안 되는가

100개 요청 중 95개가 100 ms, 5개가 5000 ms라면 평균은 345 ms입니다. 나쁘지 않아 보입니다. 하지만 사용자 20명 중 1명은 5초를 기다립니다. **평균은 그 5명을 숨깁니다.**

그래서 이 스터디는 항상 **mean, p50, p95, p99를 함께 기록**합니다.

### Latency와 Throughput은 왜 충돌하는가

Throughput을 늘리는 가장 쉬운 방법은 batch를 키우는 것입니다. 그런데 batch를 키우면:

- 요청들이 batch에 모이기를 기다려야 할 수 있고,
- 한 step에 처리하는 일이 많아져 step 시간이 조금씩 늘고,
- 대기열이 길어져 tail latency가 급격히 나빠집니다.

**"Throughput을 높이면 latency는 항상 나빠질까?"** — 항상은 아닙니다. GPU가 놀고 있는 구간에서는 batch를 키워도 latency가 거의 변하지 않고 throughput만 오릅니다. 그 "공짜 구간"이 어디까지인지를 찾는 것이 Session 4의 목표입니다.

---

## §1.6 Hands-on — Transformers Baseline

### 목표

**Inference engine이 없는 상태**의 기준선을 만듭니다. 이 숫자가 있어야 Session 2에서 SGLang이 무엇을 개선했는지 말할 수 있습니다.

### 고정 변수 (스터디 원칙 3.3)

```text
GPU              : (RunPod에서 배정받은 것 — 기록해 둘 것)
model            : Qwen/Qwen2.5-0.5B-Instruct
dtype            : bfloat16
input length     : 512 tokens (동일한 dummy prompt)
output length    : 128 tokens (min = max, EOS 무시)
sampling         : greedy (do_sample=False)
warm-up          : 1회 (측정에서 제외)
repeat           : 3회 후 중앙값 사용
```

### 변경 변수

```text
Experiment A : batch = 1
Experiment B : batch = 4
Experiment C : batch = 8
```

### 기록할 것

```text
input tokens, output tokens, total latency (s),
tokens/sec, peak GPU memory (GiB)
```

### 코드 (`session01/baseline.py`)

```python
import time, json, argparse, csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run(model, tok, batch, in_len, out_len, device):
    # 동일 길이 입력을 강제로 만든다 (padding 변수 제거)
    input_ids = torch.randint(100, 20000, (batch, in_len), device=device)
    attn = torch.ones_like(input_ids)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        min_new_tokens=out_len,     # EOS로 조기 종료 방지
        max_new_tokens=out_len,
        do_sample=False,            # greedy — 재현성 확보
        use_cache=True,             # KV Cache ON
    )
    torch.cuda.synchronize()        # ★ 없으면 GPU가 끝나기 전에 시간을 잰다
    t1 = time.perf_counter()

    latency = t1 - t0
    gen_tokens = batch * out_len
    return {
        "batch": batch,
        "input_tokens": batch * in_len,
        "output_tokens": gen_tokens,
        "latency_s": round(latency, 4),
        "tokens_per_s": round(gen_tokens / latency, 2),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--in-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--batches", default="1,4,8")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device).eval()

    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        run(model, tok, b, args.in_len, 8, device)     # warm-up (측정 제외)
        torch.cuda.reset_peak_memory_stats()
        trials = [run(model, tok, b, args.in_len, args.out_len, device)
                  for _ in range(args.repeat)]
        trials.sort(key=lambda r: r["latency_s"])
        best = trials[len(trials)//2]                  # 중앙값
        print(json.dumps(best))
        rows.append(best)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
```

### 측정할 때 반드시 지킬 것

| 항목 | 이유 |
|---|---|
| `torch.cuda.synchronize()` | CUDA는 비동기다. 없으면 "커널을 큐에 넣는 시간"을 잰다. |
| Warm-up 1회 | 첫 실행에는 커널 컴파일·메모리 할당·캐시 워밍이 섞인다. |
| `min_new_tokens = max_new_tokens` | 요청마다 output 길이가 다르면 비교가 무의미해진다. |
| `do_sample=False` | 재현 가능한 결과. |
| `reset_peak_memory_stats()` | 이전 실험의 peak가 남아 있으면 메모리 수치가 오염된다. |
| 모델 로딩 시간 제외 | 우리가 재고 싶은 건 inference지 startup이 아니다. |

### 추가 관찰 — Prefill과 Decode 분리해서 보기

여유가 있다면 다음을 추가로 측정합니다. Session 1의 개념을 눈으로 확인하는 가장 좋은 방법입니다.

```text
1) out_len = 1  로 실행       → 거의 prefill 시간
2) out_len = 128 로 실행      → prefill + decode 128회
3) (2 - 1) / 127              → step당 decode 시간 ≈ TPOT
```

그리고 `in_len`을 128 / 512 / 2048로 바꿔 가며 위를 반복하면,

- **prefill 시간은 input length에 따라 크게 증가**하고
- **decode step 시간은 거의 변하지 않는다**

는 것을 확인할 수 있습니다.

### 결과 표 템플릿

| Exp | batch | in_len | out_len | latency (s) | tokens/s | peak mem (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 512 | 128 | | | |
| B | 4 | 512 | 128 | | | |
| C | 8 | 512 | 128 | | | |

### 그래프 (1개)

```text
x축: batch size (1, 4, 8)
y축(좌): total throughput (tokens/s)     ← 증가할 것
y축(우): request 1개의 E2E latency (s)   ← 증가하거나 거의 그대로일 것
```

---

## §1.7 토론 질문

강의 후 조원과 함께 이야기합니다. **먼저 예상을 말하고, 그다음 측정값을 봅니다.**

**Q1. Batch를 키우면 항상 한 request의 latency가 좋아질까?**

<details>
<summary>논의 가이드</summary>

아니오. Batch는 개별 request를 **빠르게 만들지 않습니다.** 같은 시간 안에 더 많은 request를 처리할 뿐입니다. 오히려 batch 안의 각 request는 자기 혼자였을 때보다 조금 느려지거나 (step 시간 증가), 같거나 (memory-bound 구간에서), 배치가 채워지기를 기다렸다면 훨씬 느려집니다.
</details>

**Q2. 왜 throughput은 증가하는데 latency는 증가할 수 있을까?**

<details>
<summary>논의 가이드</summary>

Decode에서 weight 읽기 비용은 batch가 커져도 거의 동일합니다(§1.4). 그래서 batch 8의 step 시간은 batch 1의 8배가 아니라 1.1~1.5배 정도입니다. 총 처리량은 8배 가까이 늘지만, 각 request의 step 시간은 조금 늘어난 그 값이 됩니다. → throughput ↑, per-request latency도 소폭 ↑.
</details>

**Q3. Prompt가 길어질 때와 output이 길어질 때 병목은 동일할까?**

<details>
<summary>논의 가이드</summary>

다릅니다. Prompt ↑ → prefill 연산량 ↑ → **compute-bound**, TTFT에 영향. Output ↑ → decode step 횟수 ↑ → **memory-bound**, E2E와 TPOT에 영향. 게다가 둘 다 KV Cache를 키우지만 증가 속도와 지속 시간이 다릅니다.
</details>

**Q4. `use_cache=False`로 바꾸면 결과가 어떻게 변할까? 왜?**

**Q5. 우리 실험에서 batch 8일 때 GPU utilization을 `nvidia-smi`로 보면 몇 %일 것 같은가? 그 숫자가 낮다면 그건 나쁜 것인가?**

---

## §1.8 Deliverable

```text
session01-baseline/
├── README.md          # 관찰 5줄 이내 + 환경 기록
├── baseline.py
├── results.csv
└── figures/
    └── batch_vs_throughput.png
```

`README.md`의 observation은 스터디 규칙(§10)에 따라 네 문장으로 씁니다.

```text
I expected  : batch를 8배 키우면 latency도 비슷하게 늘 것이라 예상했다.
I changed   : batch size만 1 → 4 → 8로 변경했다 (나머지 조건 고정).
I measured  : E2E latency, output tokens/s, peak GPU memory.
I found     : latency는 1.0s → 1.3s로만 늘었고 throughput은 5.4배 증가했다.
              decode가 memory-bound라는 설명과 일치한다.
```

**제출 전 체크리스트**

- [ ] GPU 모델명, driver/CUDA 버전, 모델명이 README에 적혀 있다
- [ ] 실행 명령어가 그대로 복사해서 재현 가능하다
- [ ] `torch.cuda.synchronize()`가 들어 있다
- [ ] warm-up이 측정에서 제외되었다
- [ ] GPU 사용 기록 (날짜/시작/종료/예상 비용)을 남겼다

---

---

# Session 2. Inside an Inference Engine — SGLang

## 진행 순서 (권장 2시간)

| 시간 | 내용 |
|---|---|
| 0:00–0:10 | Session 1 결과 공유 (조별 1분씩) |
| 0:10–0:45 | 개념 강의 §2.1–§2.3 |
| 0:45–1:30 | Hands-on 1, 2 (GPU ON) |
| 1:30–1:50 | Code reading |
| 1:50–2:00 | Architecture diagram 그리기 |

---

## §2.1 `model.generate()`로는 왜 안 되는가

Session 1의 baseline은 잘 동작했습니다. 그런데 그걸 그대로 서비스에 올리면 어떤 일이 생길까요?

| 문제 | `generate()`에서 벌어지는 일 |
|---|---|
| **요청이 동시에 온다** | batch를 만들려면 요청을 모아야 하는데, 모으는 동안 먼저 온 요청은 그냥 기다린다. |
| **요청마다 길이가 다르다** | batch 안에서 가장 긴 요청이 끝날 때까지 나머지 슬롯은 padding을 계산한다. GPU 낭비. |
| **요청이 도중에 들어온다** | 진행 중인 batch에 끼워 넣을 수 없다. 다음 batch를 기다려야 한다. |
| **메모리 관리가 없다** | 몇 개까지 동시에 받을 수 있는지 아무도 모른다 → OOM으로 서버가 죽는다. |
| **재사용이 없다** | 100명이 같은 system prompt를 보내도 100번 다시 계산한다. |
| **streaming이 없다** | 사용자는 128 token이 다 만들어질 때까지 빈 화면을 본다. |
| **취소가 없다** | 사용자가 탭을 닫아도 GPU는 계속 생성한다. |

**Inference engine이란 결국 이 문제들을 푸는 소프트웨어입니다.** 모델은 그대로입니다. 달라지는 건 "모델을 어떻게 먹여 살리느냐"입니다.

---

## §2.2 Inference Server의 구조

```text
Client
  │   HTTP / OpenAI-compatible API
  ▼
Tokenizer                     텍스트 → token id (그리고 출력에서는 반대로)
  │
  ▼
Request Queue                 아직 실행되지 않은 요청들 (waiting)
  │
  ▼
Scheduler          ★          매 step마다 "이번에 누구를 처리할지" 결정
  │                           - 메모리가 충분한가?
  │                           - token budget을 넘지 않는가?
  │                           - prefill을 먼저? decode를 먼저?
  ▼
Batch                         이번 step에 GPU로 보낼 요청들의 묶음
  │
  ▼
Model Runner                  실제 forward pass 실행
  │                           (attention backend, CUDA Graph 등이 여기)
  ▼
KV Cache / Memory Pool  ★     KV를 어디에 저장하고, 언제 버리고, 언제 재사용할지
  │
  ▼
GPU
```

★ 표시된 두 컴포넌트가 이 스터디의 나머지 대부분을 차지합니다. (Scheduler → Session 4, KV Cache → Session 3)

### 각 컴포넌트가 어떤 metric에 영향을 주는가

| 컴포넌트 | 주로 영향을 주는 metric |
|---|---|
| Tokenizer | TTFT (아주 작지만, CPU 병목이 될 수 있음) |
| Request Queue | **TTFT, 특히 p95/p99** |
| Scheduler | TTFT, ITL, throughput, tail latency 전부 |
| Batch 구성 | throughput ↔ latency trade-off |
| KV Cache 관리 | 동시 처리 가능 요청 수, TTFT (prefix 재사용 시) |
| Model Runner | ITL, TPOT |

이 표를 Session 8에서 다시 채워 보게 됩니다.

---

## §2.3 Request Lifecycle — 요청 하나를 끝까지 따라가기

```text
1.  Client가 HTTP POST로 prompt를 보낸다
2.  서버가 요청을 받아 Request 객체를 만든다
3.  Tokenizer가 텍스트를 token id 배열로 바꾼다
4.  Request가 waiting queue에 들어간다
5.  Scheduler가 이번 step의 batch를 고른다
      - KV Cache에 이 요청의 자리가 있는가?
      - 이미 계산된 prefix가 캐시에 있는가?  ← Session 3
6.  Prefill 실행 → K,V를 cache에 기록 → 첫 token 생성
      ★ 이 시점이 TTFT
7.  첫 token을 client로 stream
8.  Request가 running 상태로 이동
9.  이후 매 step마다 다른 running 요청들과 함께 decode 1 token
      ★ 매 token 사이 간격이 ITL
10. EOS 또는 max_tokens 도달 → 완료
11. KV Cache 슬롯 해제 (또는 prefix cache로 보존)  ← Session 3
12. Client 연결 종료
```

> 📌 **여기서 중요한 관찰:** 6번과 9번 사이에 **다른 요청들이 끼어듭니다.** Session 1의 `generate()`에서는 없던 일입니다. "내 요청의 latency"는 이제 나 혼자 결정하는 것이 아니라 **서버 전체 상태의 함수**가 됩니다. 이것이 production inference의 본질적인 어려움입니다.

### Continuous Batching 미리보기 (자세히는 Session 4)

```text
Static batching:
step →  A A A A
        B B B B B B B B
        C C C ─ ─ ─ ─ ─     ← C는 끝났는데 슬롯이 비어 놀고 있다
        D D D D D D D D D D

Continuous batching:
step →  A A A A
        B B B B B B B B
        C C C E E E E E     ← C가 끝나자마자 새 요청 E가 들어온다
        D D D D D D D D D D
```

지금은 "요청이 끝나면 즉시 새 요청이 그 자리를 채운다"는 것만 이해하면 충분합니다.

---

## §2.4 Hands-on 1 — First SGLang Server

> ⚠️ **버전 주의:** SGLang은 빠르게 바뀝니다. 아래 flag가 동작하지 않으면
> `python -m sglang.launch_server --help`로 확인하고, README에 사용한 버전
> (`pip show sglang`)을 반드시 기록하세요.

### 서버 실행

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16
```

서버가 뜨면 로그를 **꼭 읽으세요.** 다음 정보가 나옵니다.

```text
- 모델 로딩 시간과 weight 메모리
- KV Cache pool 크기 (몇 개의 token을 저장할 수 있는가)   ★ §1.3에서 계산한 값과 비교!
- max running requests
- attention backend 이름
```

> 📌 §1.3에서 손으로 계산한 "내 GPU에 몇 token이 들어가는가"와 서버가 실제로 잡은 값을 비교해 보세요. 차이가 있다면 왜일까요? (힌트: `--mem-fraction-static`, CUDA Graph 버퍼, activation 공간)

### 확인 1 — Single request

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain KV cache in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

### 확인 2 — Streaming

```python
from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

t0 = time.perf_counter()
first = None
prev = None
itls = []

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Count from 1 to 50."}],
    max_tokens=128, temperature=0, stream=True,
)

for chunk in stream:
    if not chunk.choices or not chunk.choices[0].delta.content:
        continue
    now = time.perf_counter()
    if first is None:
        first = now - t0                      # ★ TTFT
    else:
        itls.append(now - prev)               # ★ ITL
    prev = now

print(f"TTFT = {first*1000:.1f} ms")
print(f"mean ITL = {sum(itls)/len(itls)*1000:.1f} ms")
```

**이 코드가 Session 1과 Session 2를 잇는 다리입니다.** Session 1에서는 E2E latency밖에 잴 수 없었지만, streaming이 있으면 TTFT와 ITL을 **분리해서** 측정할 수 있습니다. 그리고 그것이 prefill과 decode를 분리해서 보는 것과 정확히 같은 이야기입니다.

### 확인 3 — Concurrent requests

```python
import asyncio, time
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

async def one(i):
    t0 = time.perf_counter()
    first = None
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Write about topic {i}."}],
        max_tokens=128, temperature=0, stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content and first is None:
            first = time.perf_counter() - t0
    return {"ttft": first, "e2e": time.perf_counter() - t0}

async def main(n):
    t0 = time.perf_counter()
    rs = await asyncio.gather(*[one(i) for i in range(n)])
    wall = time.perf_counter() - t0
    ttfts = sorted(r["ttft"] for r in rs)
    print(f"n={n}  wall={wall:.2f}s  "
          f"TTFT p50={ttfts[len(ttfts)//2]*1000:.0f}ms  "
          f"p95={ttfts[int(len(ttfts)*0.95)]*1000:.0f}ms  "
          f"throughput={n*128/wall:.0f} tok/s")

asyncio.run(main(1))
asyncio.run(main(8))
```

n=1과 n=8을 비교하면서 **서버 로그**를 함께 보세요. running request 수와 token usage가 실시간으로 변하는 것이 보입니다. 그 로그가 곧 scheduler의 동작입니다.

---

## §2.5 Hands-on 2 — Baseline Comparison

Session 1의 Transformers baseline과 **최대한 같은 workload**로 SGLang을 돌립니다.

### 공정 비교를 위한 원칙

| 맞춰야 할 것 | 주의 |
|---|---|
| 같은 GPU, 같은 모델, 같은 dtype | 다르면 비교 자체가 무의미 |
| 같은 input/output length | SGLang도 `ignore_eos`로 길이 고정 |
| greedy sampling | 양쪽 모두 temperature=0 |
| 모델 로딩 / 서버 startup 제외 | inference만 측정 |
| warm-up 후 측정 | SGLang은 첫 요청에서 CUDA Graph 캡처 등이 일어남 |
| **cache 상태 명시** | 반복 실행 시 prefix cache가 켜져 있으면 두 번째부터 빨라짐 → §2.6 참고 |

### 결과 표

| Engine | Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | Throughput (tok/s) | Peak Memory (GiB) |
|---|---:|---:|---:|---:|---:|
| Transformers | 1 | | | | |
| SGLang | 1 | | | | |
| SGLang | 8 | | | | |

> Transformers baseline에서는 streaming이 없으므로 TTFT를 직접 재기 어렵습니다. `max_new_tokens=1`로 측정한 값을 TTFT의 근사치로 쓰고, **그 사실을 표에 각주로 남기세요.** 측정의 한계를 명시하는 것도 이 스터디의 일부입니다.

### 내장 benchmark 도구 (선택)

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 64 --max-concurrency 8
```

이 도구를 쓰더라도 **한 번은 직접 만든 스크립트로도 재보세요.** 내 손으로 TTFT를 재본 사람만 남의 benchmark 숫자를 의심할 수 있습니다.

### 논의 포인트 — "누가 빠른가"가 아니라 "왜 다른가"

concurrency 1에서는 SGLang이 Transformers보다 그다지 빠르지 않을 수도 있습니다. **그건 실패가 아닙니다.** 오히려 정확한 관찰입니다.

- Concurrency 1 → 배칭할 것이 없다 → engine의 이점 대부분이 발동하지 않는다
- Concurrency 8 → 여기서 차이가 벌어진다 → **차이를 만든 것은 모델이 아니라 스케줄링과 메모리 관리다**

이 결론을 표의 숫자로 뒷받침할 수 있으면 Session 2는 성공입니다.

---

## §2.6 관찰 실험 (5분, 저비용)

Session 3의 예고편입니다. 동일한 긴 prompt를 **연속으로 두 번** 보내 보세요.

```text
1회차 TTFT = ?
2회차 TTFT = ?   ← 왜 더 빠른가?
```

그리고 `--disable-radix-cache` 옵션으로 서버를 다시 띄워 같은 실험을 반복합니다.

> **관찰한 것을 한 문장으로 적어 두세요.** Session 3은 이 한 문장에서 시작합니다.

---

## §2.7 Code Reading

전체 SGLang 코드베이스를 읽으려 하지 마세요. **큰 흐름만** 봅니다. `mini-sglang`처럼 작은 구현이 있다면 그것을 먼저 봅니다.

찾아야 할 것은 파일 이름이 아니라 **다음 네 개의 지점**입니다.

1. **요청이 들어와서 queue에 들어가는 곳** — Request 객체에는 어떤 필드가 있는가?
2. **매 step에서 batch를 고르는 곳** — 무슨 조건으로 요청을 batch에 넣는가? 무슨 조건으로 미루는가?
3. **KV Cache 슬롯을 할당/해제하는 곳** — 자리가 없으면 어떻게 되는가?
4. **forward를 호출하는 곳** — prefill과 decode가 코드에서 어떻게 나뉘어 있는가?

각각에 대해 **함수 이름 하나와 한 줄 설명**만 노트에 적으면 충분합니다.

> 💡 코드가 너무 복잡하면 반대로 하세요. 서버 로그에 print를 추가하거나, 로그 문자열을 grep해서 그 문자열이 있는 파일부터 여세요. **동작하는 시스템에서 로그는 가장 좋은 목차입니다.**

---

## §2.8 Deliverable

```text
session02-sglang/
├── README.md
├── client_stream.py        # TTFT / ITL 측정
├── client_concurrent.py    # concurrency 1, 8
├── results.csv
├── figures/
│   └── engine_comparison.png
└── architecture.md         # 또는 이미지
```

### Architecture Diagram 요구사항

**"Request 하나가 들어와 token 하나가 나오기까지"** 를 자신의 그림으로 그립니다. 문서의 다이어그램을 복사하는 것이 아니라, **직접 실행해 본 경험을 반영**해야 합니다.

포함할 것:

- [ ] 각 단계에서 데이터가 무엇으로 변하는가 (text → token id → hidden state → logits → token id → text)
- [ ] TTFT가 측정되는 지점을 화살표로 표시
- [ ] ITL이 측정되는 지점을 표시
- [ ] KV Cache가 **쓰이는** 지점과 **읽히는** 지점을 구분
- [ ] 다른 요청이 끼어들 수 있는 지점을 표시  ← 이게 핵심

### 제출 전 체크리스트

- [ ] SGLang 버전과 실행 명령어가 그대로 적혀 있다
- [ ] Transformers baseline과 SGLang의 workload가 동일함을 명시했다
- [ ] 비교가 불완전한 부분(예: TTFT 근사치)을 limitation으로 적었다
- [ ] **GPU를 종료했다** ⚠️
- [ ] GPU 사용 기록을 남겼다

---

---

# 부록

## A. Week 1 용어집

| 용어 | 한 줄 정의 |
|---|---|
| Prefill | Prompt 전체를 한 번의 forward로 처리해 KV Cache를 채우고 첫 token을 만드는 단계 |
| Decode | Token 1개씩 반복 생성하는 단계 |
| KV Cache | 이전 token들의 Key/Value를 저장해 재계산을 피하는 메모리 |
| GQA | 여러 query head가 KV head를 공유해 KV Cache를 줄이는 구조 |
| TTFT | 요청 전송부터 첫 token 도착까지의 시간 |
| ITL | 연속한 출력 token 사이의 시간 |
| TPOT | 첫 token 이후 출력 token 1개당 평균 시간 |
| Arithmetic Intensity | FLOPs / 이동한 바이트. compute-bound인지 memory-bound인지 판별 |
| Continuous Batching | 요청이 끝나면 즉시 새 요청으로 batch 슬롯을 채우는 방식 |
| Head-of-line blocking | 앞선 무거운 요청이 뒤의 가벼운 요청을 지연시키는 현상 |

## B. 결과 공유 4문장 템플릿

```text
I expected ...
I changed  ...
I measured ...
I found    ...
```

## C. GPU 사용 기록 양식

```text
| Date | GPU | Session | Experiment | Start | End | Approx. Cost |
|------|-----|---------|------------|-------|-----|--------------|
```

## D. Week 1 총 예산

| Session | 권장 Budget | 실제 사용 |
|---|---:|---:|
| Session 1 | $0.8 | |
| Session 2 | $1.0 | |
| **Week 1 소계** | **$1.8** | |

## E. Week 2 예고

Session 2에서 "같은 prompt를 두 번 보내면 두 번째가 빠르다"는 것을 관찰했습니다. Week 2는 그 현상의 정체(**KV Cache와 RadixAttention**)와, 여러 요청을 어떻게 GPU에 밀어 넣을지(**Continuous Batching & Scheduling**)를 다룹니다.

Week 1에서 만든 두 가지가 계속 쓰입니다.

- §1.3에서 손으로 계산한 **token당 KV Cache 크기**
- §2.4에서 만든 **TTFT/ITL 측정 클라이언트**

---

> **Build less. Measure more. Understand why.**
