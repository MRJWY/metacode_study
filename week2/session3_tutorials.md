# SGLang 스케줄러 실습 가이드
> Admission Control · Retraction · Chunked Prefill 실측 실습
> (1.5~2시간 / RunPod / 저토큰 예산)

Session 2 실습(`tutorials.md`)이 **캐시 히트**를 관찰했다면,
이번에는 **스케줄러가 배치를 어떻게 만들고 부수는지**를 관찰한다.

관찰 수단이 하나 바뀐다. Session 2에서는 응답의 `meta_info.cached_tokens`로 충분했지만,
스케줄링은 **요청 단위가 아니라 이터레이션 단위**로 일어나므로 이번에는 **서버 로그**가
주 계측 대상이다.

---

## 공통 준비 (약 15분)

### 0-1. 서버 실행 — 로그를 반드시 파일로 남길 것

```bash
pip install "sglang[all]"

# 로그를 화면에도 띄우고 파일에도 남긴다 (tee). 이번 실습의 데이터 소스다.
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --port 30000 \
  --log-level info \
  2>&1 | tee server_baseline.log
```

실습마다 **로그 파일 이름을 바꿔가며** 저장하자. 나중에 서로 비교해야 한다.

```
server_baseline.log     기본 설정
server_tight.log        --mem-fraction-static 0.60
server_lpm.log          --schedule-policy lpm
server_fcfs.log         --schedule-policy fcfs
server_noradix.log      --disable-radix-cache
...
```

### 0-2. 오늘의 관찰 대상 로그 3종

```
Prefill batch. #new-seq: 5, #new-token: 1234, #cached-token: 4096, token usage: 0.31, #running-req: 12, #queue-req: 40
Decode batch. #running-req: 17, #token: 8931, token usage: 0.44, gen throughput (token/s): 1820, #queue-req: 40
KV cache pool is full. Retract requests. #retracted_reqs: 8, #new_token_ratio: 0.1043 -> 0.4312
```

| 필드 | 의미 | 어느 코드에서 나오나 |
|---|---|---|
| `#new-seq` | 이번 이터레이션에 **승인된** 요청 수 | `len(adder.can_run_list)` |
| `#new-token` | 실제로 계산한 프리필 토큰 | `adder.log_input_tokens` |
| `#cached-token` | radix tree가 공짜로 공급한 토큰 | `adder.log_hit_tokens` |
| `token usage` | KV 풀 점유율 | `1 - available_size()/total` |
| `#queue-req` | 대기 큐에 남은 요청 수 | `len(waiting_queue)` |
| `#retracted_reqs` | 쫓겨난 실행 중 요청 수 | `retract_decode()` 반환값 |
| `#new_token_ratio: a -> b` | AIMD 컨트롤러 리셋 | `a < b`면 비관적으로 되돌림 |

> **핵심 감각**: `#new-seq`는 "어드미션이 어디서 잘렸는가"이고,
> `#queue-req`는 "얼마나 밀렸는가"이며, `Retract`는 "과승인했다"는 자백이다.

### 0-3. 로그 파서 (`lab_sched.py`)

Session 2의 `lab_common.py`는 그대로 두고, 아래 파일을 추가로 저장하자.

```python
# lab_sched.py — SGLang 스케줄러 로그 파서
import re

_KV = re.compile(r"([#A-Za-z][\w\-\s()/]*?):\s*(-?[\d.]+)")


def parse_line(line: str):
    """로그 한 줄 -> (kind, fields) 또는 None"""
    if "Prefill batch" in line:
        kind = "prefill"
    elif "Decode batch" in line:
        kind = "decode"
    elif "Retract" in line:
        kind = "retract"
    else:
        return None

    fields = {}
    for k, v in _KV.findall(line):
        k = k.strip()
        fields[k] = float(v) if "." in v else int(v)

    if kind == "retract":
        m = re.search(r"new_token_ratio:\s*([\d.]+)\s*->\s*([\d.]+)", line)
        if m:
            fields["ratio_before"] = float(m.group(1))
            fields["ratio_after"] = float(m.group(2))
    return kind, fields


def parse_file(path):
    out = {"prefill": [], "decode": [], "retract": []}
    with open(path, errors="ignore") as f:
        for line in f:
            r = parse_line(line)
            if r:
                out[r[0]].append(r[1])
    return out


def summarize(path):
    """실습 리포트에 그대로 붙일 수 있는 요약"""
    d = parse_file(path)
    p, dec, r = d["prefill"], d["decode"], d["retract"]

    def s(rows, key):
        return sum(x.get(key, 0) for x in rows)

    new_tok = s(p, "#new-token")
    cached = s(p, "#cached-token")
    total = new_tok + cached

    print(f"=== {path} ===")
    print(f"프리필 배치 수      : {len(p)}")
    print(f"디코드 배치 수      : {len(dec)}")
    print(f"총 #new-token       : {new_tok}")
    print(f"총 #cached-token    : {cached}")
    print(f"캐시 히트율         : {cached/total:.1%}" if total else "캐시 히트율: n/a")
    if p:
        seqs = [x.get("#new-seq", 0) for x in p]
        print(f"#new-seq  평균/최대 : {sum(seqs)/len(seqs):.2f} / {max(seqs)}")
        qs = [x.get("#queue-req", 0) for x in p]
        print(f"#queue-req 최대     : {max(qs)}")
        us = [x.get("token usage", 0) for x in p]
        print(f"token usage 최대    : {max(us):.2f}")
    print(f"리트랙션 발생 횟수  : {len(r)}")
    if r:
        print(f"총 쫓겨난 요청 수   : {s(r, '#retracted_reqs')}")
        print(f"ratio 최대 리셋값   : {max(x.get('ratio_after', 0) for x in r):.4f}")


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        summarize(path)
        print()
```

사용법:

```bash
python lab_sched.py server_lpm.log server_fcfs.log
```

### 0-4. 부하 생성기 (`lab_load.py`)

모든 실습에서 재사용한다.

```python
# lab_load.py
import asyncio, aiohttp, time, random, string

URL = "http://localhost:30000/generate"


def salt(k=32):
    return "".join(random.choices(string.ascii_letters, k=k))


def make_prompt(n_words, shared_prefix="", tag=""):
    """n_words 개 단어짜리 고유 프롬프트. shared_prefix가 있으면 앞에 붙인다."""
    body = " ".join(f"w{j}" for j in range(n_words))
    return f"{shared_prefix}[{salt()}] {body}\n{tag}\nAnswer:"


async def one(session, prompt, max_new_tokens=16):
    t0 = time.perf_counter()
    async with session.post(URL, json={
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    }) as r:
        js = await r.json()
    meta = js.get("meta_info", {})
    return {
        "latency": time.perf_counter() - t0,
        "cached": meta.get("cached_tokens", 0),
        "prompt": meta.get("prompt_tokens", 0),
    }


async def fire(prompts, concurrency=None, max_new_tokens=16):
    """prompts를 (선택적으로 동시성 제한을 걸어) 투척하고 결과 리스트 반환"""
    sem = asyncio.Semaphore(concurrency or len(prompts))

    async def guarded(session, p):
        async with sem:
            return await one(session, p, max_new_tokens)

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as s:
        res = await asyncio.gather(*[guarded(s, p) for p in prompts])
    total = time.perf_counter() - t0

    print(f"n={len(prompts)}  total={total:.2f}s  "
          f"throughput={len(prompts)/total:.2f} req/s  "
          f"avg_latency={sum(r['latency'] for r in res)/len(res)*1000:.0f}ms  "
          f"cached_sum={sum(r['cached'] for r in res)}")
    return res
```

### 0-5. 유용한 엔드포인트

```bash
curl -X POST http://localhost:30000/flush_cache    # 캐시 초기화 (대조 실험 필수)
curl http://localhost:30000/get_server_info        # max_running_requests, KV 풀 크기 등
```

`get_server_info`의 출력에서 **KV 풀 총 토큰 수**를 반드시 메모해 두자.
실습 1과 3에서 예산 계산을 손으로 검증할 때 쓴다.

> ⚠️ 서버를 재시작할 때마다 KV 풀 크기가 달라질 수 있다(`--mem-fraction-static` 변경 시).
> 실습마다 다시 확인할 것.

---

## 실습 1 — 어드미션 컨트롤 관찰: `#new-seq`는 언제 잘리는가 (약 25분) ⭐

**목표**: 같은 요청을 던져도 **토큰 예산이 좁아지면 한 배치에 승인되는 요청 수가 줄어드는 것**을
로그로 확인한다. 코드로는 `PrefillAdder.add_one_req`의 `NO_TOKEN` 반환과
어드미션 루프의 `break`를 눈으로 보는 실습이다.

### 1-1. 실험 설계

프롬프트를 **일부러 길게(약 800단어)** 만들고, 캐시 히트가 섞이지 않도록 **전부 고유하게**
만든다. 그래야 순수하게 "예산" 효과만 본다.

```python
# lab1_admission.py
import asyncio
from lab_load import make_prompt, fire

prompts = [make_prompt(800, tag=f"Question {i}?") for i in range(24)]
asyncio.run(fire(prompts, max_new_tokens=16))
```

### 1-2. 실행 절차

**(A) 넉넉한 예산**

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --port 30000 --log-level info --mem-fraction-static 0.85 \
  2>&1 | tee server_loose.log
# 다른 터미널에서
python lab1_admission.py
```

**(B) 좁은 예산** — 서버를 끄고 다시:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --port 30000 --log-level info --mem-fraction-static 0.60 \
  2>&1 | tee server_tight.log
python lab1_admission.py
```

**(C) 요청 수 상한으로 자르기**

```bash
python -m sglang.launch_server ... --max-running-requests 4 \
  2>&1 | tee server_maxrun4.log
python lab1_admission.py
```

### 1-3. 관찰 및 집계

```bash
python lab_sched.py server_loose.log server_tight.log server_maxrun4.log
```

| 로그 | `#new-seq` 평균 | `#new-seq` 최대 | `#queue-req` 최대 | 프리필 배치 수 |
|---|---|---|---|---|
| loose (0.85) | | | | |
| tight (0.60) | | | | |
| maxrun4 | | | | |

### 1-4. 예상 결과와 해석

- **loose**: 24개가 1~2번의 프리필 배치에 거의 다 들어간다. `#new-seq`가 크다.
- **tight**: `#new-seq`가 3~6 수준으로 잘리고 프리필 배치 수가 늘어난다.
  **같은 요청, 같은 정책인데 예산만 달라서 배치 구성이 달라졌다.**
- **maxrun4**: `#new-seq`가 4를 절대 못 넘는다.
  이건 예산이 아니라 `running_bs >= self.max_running_requests` 가드에 걸린 것이다.

**중요한 구분** — tight와 maxrun4는 겉보기에 비슷하지만 원인이 다르다:

| | 잘리는 이유 | 코드 위치 | `token usage` |
|---|---|---|---|
| tight | KV 토큰 부족 (`NO_TOKEN`) | `add_one_req` (1)번 검사 | 높게 유지됨 |
| maxrun4 | 동시 요청 수 상한 | 어드미션 루프 맨 위 | **낮은데도** 잘림 |

`token usage`가 낮은데 `#queue-req`가 높다면 그건 메모리 문제가 아니라 설정 문제다.
이게 Part 2 §11 진단 치트시트의 첫 줄이다.

### 1-5. 토론 포인트

- 800단어 프롬프트가 대략 몇 토큰인지 `meta_info.prompt_tokens`로 확인하고,
  `get_server_info`의 KV 풀 크기로 나눠 보자. "한 배치에 몇 개가 들어가야 맞는가"를
  손으로 계산한 값과 `#new-seq`가 얼마나 맞는가?
- 안 맞는다면 그 차이는 어디서 오는가? (힌트: `max_new_tokens` 예약분과 `new_token_ratio`)

---

## 실습 2 — LPM vs FCFS를 스케줄러 관점에서 (약 25분) ⭐

Session 2 실습 4를 다시 하되, 이번에는 **응답의 `cached_tokens` 합계가 아니라
서버 로그의 `#new-seq`를 본다.** 같은 현상을 반대편에서 보는 것이다.

### 2-1. 실험 설계 — 콜드 요청을 맨 앞에 심는다

Part 2 §5.8 트레이스 C의 재현이다. 핵심은 **"뚱뚱한 콜드 요청을 큐 맨 앞에 두는 것"**.
FCFS는 그걸 먼저 처리하려다 예산을 다 쓰고, LPM은 뒤로 미룬다.

```python
# lab2_policy.py
import asyncio
from lab_load import make_prompt, fire, salt

SYSTEMS = [
    "You are a chef. " * 40,
    "You are a lawyer. " * 40,
    "You are a poet. " * 40,
]

# 1) 아주 뚱뚱한 콜드 요청 하나 (공유 prefix 없음)
cold = make_prompt(1500, tag="Write a long essay about numbers.")

# 2) 공유 prefix를 가진 요청 30개 (A,B,C,A,B,C,... interleave)
warm = []
for i in range(10):
    for s in SYSTEMS:
        warm.append(s + f"\nUser: question {i}\nAssistant:")

prompts = [cold] + warm     # 콜드가 맨 앞

asyncio.run(fire(prompts, max_new_tokens=16))
```

### 2-2. 실행 절차

**반드시 각 실행 전에 캐시를 비울 것.** 안 그러면 두 번째 실행이 첫 번째 덕을 본다.

```bash
# (A) LPM
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --port 30000 --log-level info \
  --schedule-policy lpm --mem-fraction-static 0.60 \
  2>&1 | tee server_lpm.log

curl -X POST http://localhost:30000/flush_cache
python lab2_policy.py

# (B) FCFS — 서버 재시작
python -m sglang.launch_server ... --schedule-policy fcfs --mem-fraction-static 0.60 \
  2>&1 | tee server_fcfs.log

curl -X POST http://localhost:30000/flush_cache
python lab2_policy.py
```

### 2-3. 집계

```bash
python lab_sched.py server_lpm.log server_fcfs.log
```

| | LPM | FCFS |
|---|---|---|
| 총 `#new-token` | | |
| 총 `#cached-token` | | |
| 캐시 히트율 | | |
| `#new-seq` 평균 | | |
| 프리필 배치 수 | | |
| 전체 소요 시간 (`fire` 출력) | | |

### 2-4. 예상 결과와 해석

LPM에서는 첫 프리필 배치의 `#new-seq`가 크고 `#cached-token`이 높게 찍히며, 뚱뚱한 콜드
요청은 뒤로 밀린다. FCFS에서는 초기 몇 이터레이션의 `#new-seq`가 작게 나온다.

**해석의 핵심**: 총 계산량은 두 정책이 비슷할 수 있다. 차이는
1. 값싼 요청들이 먼저 빠져나가 평균 대기 시간이 줄고,
2. 공유 prefix 노드가 한 번 잠긴 뒤 형제 요청들이 **예산을 거의 안 쓰고** 승인되어
   한 배치에 더 많이 들어간다는 것이다 (Part 2 §5.5의 잠금/예산 상호작용).

### 2-5. 차이가 안 보인다면

> Session 2 실습 4에도 같은 주의사항이 있었다. 메모리가 넉넉하면 FCFS에서도 `NO_TOKEN`이
> 안 나서 결국 전부 승인되고, 두 정책이 같은 결과를 낸다.

차이가 안 나면 순서대로 시도:
1. `--mem-fraction-static`을 0.55, 0.50으로 더 낮춘다
2. 콜드 요청의 단어 수를 1500 → 3000으로 늘린다
3. warm 요청 수를 30 → 60으로 늘린다

**"차이가 안 나는 조건"을 찾는 것 자체가 좋은 학습이다.** 어떤 조건에서 스케줄 정책이
무의미해지는지 아는 것이 실전에서는 더 중요하다.

---

## 실습 3 — 리트랙션 유도와 재프리필 비용 (약 30분) ⭐⭐

**목표**: `KV cache pool is full. Retract requests.` 로그를 실제로 띄우고,
리트랙션된 요청이 재진입할 때 radix tree가 얼마나 구해 주는지를 정량화한다.
Part 2 §7.3 트레이스 D의 실측 버전이다.

### 3-1. 리트랙션이 나오는 조건

리트랙션은 **축출로도 부족할 때만** 나온다. 따라서 셋 다 필요하다:

1. KV 풀이 작을 것 → `--mem-fraction-static` 낮추기
2. 생성이 길 것 → `--random-output-len` 크게 (실행 중 요청의 메모리가 계속 자람)
3. 동시성이 높을 것 → `--request-rate` 크게

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000 \
  --mem-fraction-static 0.55 \
  --max-running-requests 128 \
  --log-level info \
  2>&1 | tee server_retract.log
```

```bash
python -m sglang.bench_serving --backend sglang \
  --dataset-name random --num-prompts 128 \
  --random-input-len 1024 --random-output-len 512 \
  --request-rate 64
```

### 3-2. 리트랙션이 안 나올 때

```
mem-fraction-static:  0.55 -> 0.50 -> 0.45   (한 단계씩 낮춘다)
random-output-len:    512 -> 1024
num-prompts:          128 -> 256
max-running-requests: 128 -> 256
```

`token usage`가 0.95 이상까지 올라가는지 먼저 확인하자. 거기까지 안 가면 리트랙션은 절대
안 난다. 반대로 서버가 OOM으로 죽으면 `--mem-fraction-static`을 한 단계 되돌린다.

### 3-3. 관찰 절차

**(1) 리트랙션 로그를 찾는다**

```bash
grep -n "Retract" server_retract.log | head -20
```

```
KV cache pool is full. Retract requests. #retracted_reqs: 3, #new_token_ratio: 0.1043 -> 0.4312
```

`0.1043 -> 0.4312` — 값이 **올라간** 것을 확인하자. 이게 AIMD의 "리셋" 구간이다
(Part 1 §3.3의 그래프에서 뾰족하게 튀는 부분).

**(2) 그 직후의 프리필 로그를 본다** — 재진입 순간이다

```bash
grep -n -A 30 "Retract" server_retract.log | grep "Prefill batch" | head -5
```

`#cached-token`이 크고 `#new-token`이 작은 줄을 찾자. 그게 리트랙션된 요청이
자기 자신의 옛 prefix에 히트해서 되돌아온 것이다.

**(3) 대조군: radix cache를 끈다**

```bash
python -m sglang.launch_server ... --mem-fraction-static 0.55 \
  --disable-radix-cache 2>&1 | tee server_retract_noradix.log
# 동일한 bench_serving 재실행
```

### 3-4. 집계표

```bash
python lab_sched.py server_retract.log server_retract_noradix.log
```

| | radix ON | radix OFF |
|---|---|---|
| 리트랙션 횟수 | | |
| 총 쫓겨난 요청 수 | | |
| 총 `#new-token` | | |
| 총 `#cached-token` | | 0 (당연) |
| bench_serving 총 처리량 | | |
| bench_serving mean TTFT | | |

### 3-5. 해석

- radix ON에서 리트랙션된 요청은 **이미 생성한 토큰 수만큼만** 재계산한다.
  6000토큰 요청이 3토큰만 다시 계산하는 것도 가능하다.
- radix OFF에서는 **전부 재계산**한다. 리트랙션 1회의 비용이 수천 배 차이 난다.
- 이것이 "리트랙션은 최악의 경우 비싸지만 일반적으로는 거의 공짜"라는 말의 실체다.
  **단, tree가 그 prefix를 아직 들고 있을 때만.**

### 3-6. 토론 포인트

- `#new_token_ratio`가 리셋된 뒤 다시 내려오는 데 몇 이터레이션이 걸리는가?
  로그에서 연속된 리트랙션 사이 간격을 세어 보자. 간격이 계속 짧다면 서버는 서빙이 아니라
  **스래싱** 중이다.
- `--schedule-conservativeness 2.0`으로 올려서 재실행하면 리트랙션 횟수가 줄어드는가?
  대신 무엇이 나빠지는가? (`#new-seq`, 처리량을 보자)
- 희생자로 어떤 요청이 뽑혔을지 Part 2 §7.2의 정렬 키로 예측해 보자.
  `bench_serving`은 입력 길이가 랜덤이므로 **입력이 가장 길고 생성이 가장 적은** 요청이
  먼저 나가야 한다.

---

## 실습 4 — 청크 프리필과 ITL (약 20분)

**목표**: 긴 프롬프트 하나가 다른 요청들의 토큰 스트림을 얼마나 얼리는지 측정하고,
청크 프리필이 그걸 어떻게 완화하는지 본다.

### 4-1. 실험 설계

**배경 부하**(디코딩 중인 요청 여러 개)를 깔아 두고, 그 위에 **아주 긴 프롬프트 하나**를
던진 뒤, 배경 요청들의 지연 시간이 어떻게 변하는지 본다.

```python
# lab4_chunked.py
import asyncio, time
from lab_load import make_prompt, one
import aiohttp

async def main():
    async with aiohttp.ClientSession() as s:
        # 배경: 짧은 프롬프트 + 긴 생성 20개 (계속 디코딩하게)
        bg = [one(s, make_prompt(50, tag=f"bg{i}"), max_new_tokens=256)
              for i in range(20)]
        bg_task = asyncio.gather(*bg)

        await asyncio.sleep(1.0)          # 배경이 디코드 단계에 들어갈 때까지 대기

        # 폭탄: 아주 긴 프롬프트 1개
        t0 = time.perf_counter()
        big = await one(s, make_prompt(8000, tag="huge"), max_new_tokens=16)
        print(f"긴 프롬프트 latency = {big['latency']*1000:.0f}ms")

        res = await bg_task
        lats = sorted(r["latency"] for r in res)
        print(f"배경 요청 latency  median={lats[len(lats)//2]*1000:.0f}ms  "
              f"max={lats[-1]*1000:.0f}ms")

asyncio.run(main())
```

### 4-2. 실행 절차

```bash
# (A) 청킹 켜짐 (기본값)
python -m sglang.launch_server ... --chunked-prefill-size 2048 \
  2>&1 | tee server_chunk2048.log
python lab4_chunked.py

# (B) 청킹 꺼짐
python -m sglang.launch_server ... --chunked-prefill-size -1 \
  2>&1 | tee server_nochunk.log
python lab4_chunked.py

# (C) 작은 청크
python -m sglang.launch_server ... --chunked-prefill-size 512 \
  2>&1 | tee server_chunk512.log
python lab4_chunked.py
```

### 4-3. 관찰 포인트

로그에서 긴 프롬프트가 처리되는 구간을 찾아보자.

- **청킹 ON**: `Prefill batch. #new-token: 2048` 이 여러 번 연속으로 찍히고,
  그 사이사이에 `Decode batch.`가 계속 돈다.
- **청킹 OFF**: `Prefill batch. #new-token: 8000+` 이 **한 번** 찍히고,
  그 앞뒤로 디코드가 뚝 끊긴다.

| | 긴 프롬프트 latency | 배경 median | 배경 max |
|---|---|---|---|
| chunk 2048 | | | |
| chunk 512 | | | |
| 청킹 OFF | | | |

**예상**: 청킹 OFF에서 긴 프롬프트 자체는 조금 빠르지만(TTFT 유리), 배경 요청의 max latency가
확 튄다. 청크 512는 배경이 가장 부드럽지만 긴 프롬프트가 가장 느리다.
**이게 Part 1 §5.4의 트레이드오프 표를 숫자로 만든 것이다.**

---

## 실습 5 — 오버랩 스케줄링 (약 10분, 선택)

**목표**: CPU/GPU 파이프라이닝이 처리량에 미치는 영향을 1.5B 모델에서 측정한다.
포워드 패스가 짧은 소형 모델일수록 효과가 크다는 것이 Part 1 §6.1의 주장이었다.

```bash
# (A) 기본 (오버랩 ON)
python -m sglang.launch_server ... 2>&1 | tee server_overlap.log
python -m sglang.bench_serving --backend sglang --dataset-name random \
  --num-prompts 64 --random-input-len 256 --random-output-len 64 --request-rate 16

# (B) 오버랩 OFF
python -m sglang.launch_server ... --disable-overlap-schedule \
  2>&1 | tee server_nooverlap.log
# 동일 벤치마크
```

`gen throughput (token/s)` 필드의 평균을 비교하자:

```python
from lab_sched import parse_file
for p in ["server_overlap.log", "server_nooverlap.log"]:
    d = parse_file(p)["decode"]
    vals = [x["gen throughput (token/s)"] for x in d if "gen throughput (token/s)" in x]
    print(p, f"평균 {sum(vals)/len(vals):.0f} tok/s  (n={len(vals)})")
```

**토론**: 만약 7B 모델로 같은 실험을 하면 차이가 커질까 작아질까? 왜?

---

## 실습 6 — 브레이크포인트로 직접 들여다보기 (심화, 약 20분)

`--disable-overlap-schedule`로 띄워야 한다. 오버랩 모드에서는 지금 보고 있는 결과가
**직전 배치**의 것이라 처음 보면 반드시 헷갈린다.

```bash
python -m sglang.launch_server ... --disable-overlap-schedule --tp 1
```

### 6-1. 어드미션 예산이 잠금 전후로 변하는지 확인

`schedule_policy.py`의 `add_one_req` 진입부에 넣자:

```python
def add_one_req(self, req, has_chunked_req, truncation_align_size=None):
    before = self.rem_total_tokens          # <-- 추가
    total_tokens = req.extend_input_len + min(...)
    ...
    with self._lock_node(req.last_node):
        after = self.rem_total_tokens        # <-- 추가
        print(f"[ADMIT] rid={req.rid[:8]} extend={req.extend_input_len} "
              f"prefix={len(req.prefix_indices)} "
              f"rem: {before:.0f} -> {after:.0f} (delta={after-before:.0f})")
```

**확인할 것**: prefix 매칭이 큰 요청에서 `delta`가 음수로 크게 나오는가?
그게 `inc_lock_ref`가 `evictable_size_`를 `protected_size_`로 옮긴 양이다
(Part 2 §5.5의 (3)(4) 재확인이 왜 필요한지의 증거).

### 6-2. 리트랙션 정렬 키 확인

`schedule_batch.py`의 `retract_decode` 진입부:

```python
def retract_decode(self, server_args):
    keys = [(len(r.output_ids), -len(r.origin_input_ids), r.rid[:8]) for r in self.reqs]
    print("[RETRACT] 정렬 전:", sorted(keys, reverse=True))
```

**확인할 것**: 실제로 pop되는 첫 희생자가 `(생성 최소, 입력 최장)`인가?
본인 브랜치에서 정렬 키에 `priority`가 섞여 있는가?

### 6-3. 이벤트 루프 상태 덤프

`scheduler.py`의 `get_next_batch_to_run` 진입부:

```python
def get_next_batch_to_run(self):
    if self.forward_ct % 50 == 0:      # 50 이터레이션마다
        print(f"[LOOP {self.forward_ct}] queue={len(self.waiting_queue)} "
              f"running={self.running_batch.batch_size() if self.running_batch else 0} "
              f"last={self.last_batch.forward_mode if self.last_batch else None} "
              f"full={self.batch_is_full} ratio={self.new_token_ratio:.4f} "
              f"chunked={self.chunked_req is not None}")
```

**확인할 것**: `batch_is_full=True`가 몇 이터레이션 동안 유지되는가?
그동안 `queue`는 줄지 않고 `running`만 줄어드는가?

---

## 시간 배분 요약

| 순서 | 실습 | 시간 | 우선순위 |
|---|---|---|---|
| 0 | 서버 셋업 + 파서/부하생성기 | 15분 | 필수 |
| 1 | 어드미션 관찰 | 25분 | ⭐ 필수 |
| 2 | LPM vs FCFS | 25분 | ⭐ 필수 |
| 3 | 리트랙션 + 재프리필 | 30분 | ⭐⭐ 필수 |
| 4 | 청크 프리필과 ITL | 20분 | 권장 |
| 5 | 오버랩 스케줄링 | 10분 | 선택 |
| 6 | 브레이크포인트 | 20분 | 심화 |

## 토큰 절약 팁

- `max_new_tokens`는 **실습 1, 2, 4에서는 16**으로 고정. 프리필/어드미션 관찰이 목적이라
  생성은 짧아도 된다.
- **실습 3만 예외**로 생성이 길어야 한다(리트랙션은 실행 중 요청의 메모리가 자라야 발생).
  대신 `--num-prompts`를 줄여서 총량을 조절하자.
- `temperature=0` — 재현 가능한 결과. 두 정책을 비교할 때 필수다.
- 서버 재시작이 잦은 실습이다. 모델 로딩 시간을 아끼려면
  `--model-path`를 로컬 캐시 경로로 지정하자.

---

# 과제

> 제출물은 `report.md` 하나 + 로그 파일들. 아래 템플릿을 그대로 쓰면 된다.
> 각 과제마다 **"내가 예측한 것 → 실제 결과 → 차이의 원인"** 3단 구성으로 쓸 것.
> 결과가 예측과 다른 것은 감점 요소가 아니다. **차이를 설명하지 못하는 것**이 감점 요소다.

## 과제 1 — 손으로 어드미션 추적하기 (코드 실행 없음)

**난이도**: ★★☆ | **예상 시간**: 30분

### 상황

요청 4개가 **동시에** 도착했다. radix tree는 **비어 있다**.

| 요청 | 도착 순서 | 프롬프트 토큰 | `max_new_tokens` |
|---|---|---|---|
| R1 | 1 | 100 | 128 |
| R2 | 2 | 8000 | 128 |
| R3 | 3 | 150 | 128 |
| R4 | 4 | 200 | 128 |

```
KV 풀 여유 = 9000 토큰,  evictable = 0,  running_batch 비어 있음
new_token_ratio = 0.7,  max_prefill_tokens = 16384
chunked_prefill_size = 8192,  page_size = 1
```

### 수행 절차

1. `PrefillAdder.__init__`의 초기 `rem_total_tokens`를 계산하라.
2. **FCFS**로 어드미션 루프를 한 줄씩 손으로 돌려라. 각 요청마다:
   - `total_tokens = extend_input_len + min(max_new_tokens, CLIP_MAX_NEW_TOKENS)`
   - `total_tokens >= rem_total_tokens` 인가?
   - `input_tokens <= rem_chunk_tokens` 인가? (아니면 청크 절단)
   - 반환값이 `CONTINUE` / `NO_TOKEN` / `OTHER` 중 무엇인가?
   - 승인 후 `rem_total_tokens`는 얼마가 되는가?
3. 이터레이션 1에서 **어떤 요청이 승인되고 어떤 요청이 큐에 남는가?**
   `batch_is_full`은 켜지는가?
4. `#new-seq`, `#new-token`, `#cached-token` 값을 예측하라.
5. 이제 **LPM**으로 같은 계산을 하라. tree가 비어 있으니 모든 요청의 prefix 매칭이 0이다.
6. 이터레이션 2, 3에서 무슨 일이 벌어지는지 계속 추적하라 (R2는 언제 처리되는가?).

### 함정 (반드시 다뤄야 할 것)

- R2는 8000토큰인데 `chunked_prefill_size`가 8192다. 청크 절단이 일어나는가, 안 일어나는가?
  그 판정은 `add_one_req`의 어느 조건식에서 이루어지는가?
- **빈 tree에서의 LPM은 FCFS와 같은가?** 같지 않다. 왜인가?
  힌트: `_compute_prefix_matches`가 `waiting_queue_radix_tree`(시뮬레이션 tree)를 함께
  돌리며 `temporary_deprioritized`를 만든다. 이 4개 요청은 서로 prefix를 공유하는가?
  `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`가 어떤 역할을 하는가?
- `add_one_req` (1)번의 부등호는 `>` 가 아니라 `>=` 다. 경계값에서 차이가 나는가?

### 제출물

```markdown
## 과제 1

### FCFS 추적표
| 단계 | 요청 | total_tokens | rem_total_tokens (전) | 판정 | rem_total_tokens (후) |
|---|---|---|---|---|---|
| 1 | R1 | 228 | 9000 | CONTINUE | 8772 |
| ... |

### 이터레이션별 배치 구성
| 이터 | 승인된 요청 | #new-seq | #new-token | 큐에 남은 것 |
|---|---|---|---|---|

### LPM 추적표
(동일 형식)

### FCFS와 LPM이 다른가? 왜?
(3~5문장)
```

### 검증 (선택, 가산점)

실제로 서버를 띄워서 이 시나리오를 재현하고, 예측한 `#new-seq`와 로그를 비교하라.
`--mem-fraction-static`을 조절해 KV 풀을 9000토큰 근처로 맞추는 것이 관건이다
(`get_server_info`로 확인).

---

## 과제 2 — `batch_is_full` 스톨 재현하기

**난이도**: ★★★ | **예상 시간**: 45분

### 목표

**`token usage`는 0.5 이하인데 `#queue-req`가 계속 높게 유지되는** 상태를 인위적으로 만들어라.
즉 "메모리는 남는데 요청이 안 들어가는" 상황이다.

### 왜 이게 가능한가

`rem_total_tokens`는 실행 중 요청의 **미래 디코드 메모리를 미리 예약**한다:

```
rem_total_tokens = free + evictable - Σ_running(남은 max_new_tokens × new_token_ratio)
```

`max_new_tokens`를 아주 크게 잡으면 예약분이 커져서, 실제 점유율(`token usage`)이 낮은데도
신규 승인이 막힌다.

### 수행 절차

1. 서버를 띄운다. 로그를 `server_stall.log`로 남긴다.
2. **`max_new_tokens`를 아주 크게** 설정한 요청 N개를 먼저 던져 running_batch를 채운다.
   ```python
   # 힌트: max_new_tokens=4096 짜리 요청 16개를 먼저 던진다
   ```
3. 1~2초 뒤, **짧은 요청 50개**를 추가로 던진다.
4. 로그에서 `#queue-req`가 높은데 `token usage`가 낮은 구간을 찾는다.
5. 그 구간이 **몇 이터레이션 지속되는지** 센다.

### 수집할 데이터

```python
from lab_sched import parse_file
d = parse_file("server_stall.log")
for row in d["prefill"] + d["decode"]:
    if row.get("#queue-req", 0) > 10 and row.get("token usage", 1) < 0.5:
        print(row)
```

### 답해야 할 질문

1. 스톨 구간의 최대 `#queue-req`와 그때의 `token usage`는?
2. 스톨이 **풀리는 순간** 무슨 일이 일어났는가?
   (힌트: `batch_is_full = False`가 되는 두 지점을 Part 2 §7에서 찾아라)
3. 다음 셋 중 **무엇이 이 문제를 고치는가?** 각각 실험해서 답하라:
   - `--schedule-conservativeness 0.5` (예약을 덜 하게)
   - `--max-running-requests`를 낮추기
   - 클라이언트가 `max_new_tokens`를 현실적으로 설정하기
4. 실제 서비스에서 이 문제를 만나면 어느 것을 고르겠는가? 왜?

### 제출물

- 스톨 구간의 로그 발췌 (10~20줄)
- 위 3번의 세 조건에 대한 비교표 (`#queue-req` 최대, `token usage` 평균, 총 소요 시간)
- 4번에 대한 3~5문장 논증

---

## 과제 3 — 소스 읽기: full vs chunked 분기

**난이도**: ★★☆ | **예상 시간**: 20분

### 수행 절차

1. 본인 체크아웃에서 `add_one_req`를 찾는다:
   ```bash
   grep -n "def add_one_req" python/sglang/srt/managers/schedule_policy.py
   ```
2. full 프리필과 chunked 프리필을 가르는 **정확한 조건식**을 찾아 그대로 인용하라.
3. 그 조건식의 각 절(clause)이 참이 되는 상황을 하나씩 설명하라.
4. 특히 `return_logprob` 관련 절이 왜 거기 있는지 설명하라.
   힌트: prompt logprob은 프롬프트 **전체**가 한 번의 포워드 패스에 있어야 계산된다.
   청크로 쪼개지면 무엇이 불가능해지는가?
5. `chunked=True`가 `cache_unfinished_req` → `insert(InsertParams(..., chunked=chunked))`까지
   전달되는 경로를 추적하고, `_inc_hit_count`에서 이 플래그가 왜 필요한지 설명하라.
   (힌트: 청크 재방문을 "캐시 히트"로 세면 LFU 정책의 통계가 오염된다)

### 제출물

```markdown
## 과제 3
### 조건식 원문
```python
(파일:라인 명시 후 인용)
```
### 각 절의 의미
| 절 | 참이 되는 상황 | 결과 |
|---|---|---|

### return_logprob 절의 이유
(3~5문장)

### chunked 플래그 전파 경로
schedule_policy.py:XXX -> scheduler.py:XXX -> radix_cache.py:XXX -> ...
```

---

## 과제 4 — 리트랙션 정렬 키 검증

**난이도**: ★★★ | **예상 시간**: 40분

### 목표

Part 2 §7.2에서 설명한 정렬 키가 **본인 브랜치에서 실제로 그런지** 검증한다.
문서는 틀릴 수 있다. 코드는 안 틀린다.

### 수행 절차

1. `retract_decode`를 찾아 정렬 키를 그대로 인용하라.
   ```bash
   grep -n "def retract_decode" python/sglang/srt/managers/schedule_batch.py
   ```
2. 정렬 키에 `priority`가 포함되어 있는가? `reverse=True`인가? `pop()`은 어느 쪽에서 하는가?
3. 아래 계측 코드를 `retract_decode` 진입부에 삽입하라:
   ```python
   print("[RETRACT-IN] " + str(sorted(
       [(len(r.output_ids), len(r.origin_input_ids), r.rid[:8]) for r in self.reqs]
   )))
   ```
   그리고 희생자를 pop하는 자리에:
   ```python
   print(f"[RETRACT-OUT] rid={req.rid[:8]} out={len(req.output_ids)} "
         f"in={len(req.origin_input_ids)}")
   ```
4. 실습 3의 설정으로 서버를 띄워 리트랙션을 유발한다.
5. **첫 희생자가 정말 "생성 최소 + 입력 최장"인가?** 로그로 확인하라.

### 답해야 할 질문

1. 실제 정렬 키를 그대로 적어라. 문서(§7.2)와 일치하는가? 다르다면 어떻게 다른가?
2. `while ... or first_iter` 조건에서 `first_iter`가 없으면 어떤 버그가 생기는가?
3. `if len(sorted_indices) == 1: break`를 제거하면 어떤 일이 벌어지는가?
   (실제로 지우고 돌려보되, **반드시 별도 브랜치에서** 할 것)
4. `retract_decode_steps` 값을 `global_config.py`에서 찾아라. 이 값을 2로 줄이면
   리트랙션 빈도가 어떻게 변할 것 같은가? 실제로 바꿔서 확인하라.

### 제출물

- 정렬 키 원문 + 문서와의 차이
- `[RETRACT-IN]` / `[RETRACT-OUT]` 로그 발췌 3회분
- 예측과 실제가 일치하는지에 대한 판정
- 질문 2~4에 대한 답

---

## 과제 5 — LPM의 이득을 정량화하고 괴리를 설명하기

**난이도**: ★★★★ | **예상 시간**: 60분

### 목표

실습 2에서 얻은 "토큰 절감량"과 "실제 벽시계 시간 절감량" 사이의 **괴리**를 설명한다.
이게 이 과제의 핵심이다. 토큰을 77% 아꼈는데 시간은 20%밖에 안 줄었다면, 나머지는 어디로 갔나?

### 수행 절차

1. 실습 2를 각 정책마다 **3회씩** 반복 실행하고 중앙값을 쓴다 (분산이 크다).
2. 아래 지표를 모두 수집한다:

| 지표 | 출처 |
|---|---|
| 총 `#new-token` | `lab_sched.summarize` |
| 총 `#cached-token` | `lab_sched.summarize` |
| 프리필 배치 수 | `lab_sched.summarize` |
| 디코드 배치 수 | `lab_sched.summarize` |
| 전체 소요 시간 | `fire()` 출력 |
| 평균 요청 latency | `fire()` 출력 |
| 평균 `#running-req` (디코드) | 직접 계산 |

3. **토큰 절감률**과 **시간 절감률**을 각각 계산하라.
   ```
   토큰 절감률 = 1 - (LPM의 총 #new-token / FCFS의 총 #new-token)
   시간 절감률 = 1 - (LPM의 총 소요 시간 / FCFS의 총 소요 시간)
   ```

### 괴리를 설명하기 위해 반드시 고려할 것

1. **디코드가 전체 시간에서 차지하는 비중.** 프리필 토큰을 아껴도 디코드 시간은 안 줄어든다.
   `max_new_tokens=16`인 이 실험에서 디코드 배치 수 × 배치당 시간은 얼마인가?
2. **프리필은 compute-bound다.** 토큰 수를 절반으로 줄이면 시간도 절반이 되는가?
   (커널 런치 오버헤드, 배치 고정비용)
3. **평균 디코드 배치 크기.** LPM이 한 배치에 더 많이 승인했다면 `#running-req` 평균이
   높아야 한다. 실제로 그런가? 이게 처리량에 어떻게 기여하는가?
4. **`max_new_tokens`를 64, 256으로 늘리면 괴리가 커지는가 작아지는가?**
   최소 한 번은 실험해서 답하라.

### 제출물

- 3회 반복 측정 원자료 표
- 토큰 절감률 vs 시간 절감률 비교 (숫자)
- **괴리에 대한 정량적 설명**: 위 1~4를 근거로, 절감되지 않은 시간이 어디에 쓰였는지
  대략적인 수지를 맞춰 보라 (예: "토큰 절감으로 아낀 프리필 시간 X초 중 Y초는 디코드가
  차지하고, Z초는 배치 고정비용이다")
- `max_new_tokens` 변화 실험 결과

---

## 과제 6 — 토큰 하나의 일생 추적하기

**난이도**: ★★★★ | **예상 시간**: 60분

### 목표

요청 하나를 골라, 그 요청의 KV 캐시가 **언제 할당되고 언제 해제되는지**를
코드 경로로 완전히 추적한다.

### 수행 절차

1. `--disable-overlap-schedule`로 서버를 띄운다.
2. 아래 지점들에 로그를 삽입한다. **`rid`를 반드시 찍어서 한 요청을 끝까지 따라갈 수 있게 하라.**

| 지점 | 파일 | 찍을 것 |
|---|---|---|
| 큐 진입 | `scheduler.py` `_add_request_to_queue` | `rid`, `len(origin_input_ids)` |
| prefix 매칭 | `schedule_batch.py` `init_next_round_input` | `rid`, `len(prefix_indices)`, `extend_input_len` |
| 승인 | `schedule_policy.py` `add_one_req` | `rid`, 판정 결과 |
| KV 할당 | `schedule_batch.py` `prepare_for_extend` | `rid`, `req_pool_idx`, 할당 슬롯 수 |
| 첫 토큰 | `batch_result_processor.py` `process_batch_result_prefill` | `rid`, `next_token_id`, `finished()` |
| 디코드 | `schedule_batch.py` `prepare_for_decode` | 배치 크기, `seq_lens` 합 |
| 매 토큰 | `batch_result_processor.py` `process_batch_result_decode` | `rid`, `len(output_ids)` |
| 종료 | `mem_cache/common.py` `release_kv_cache` | `rid`, `effective_kv_committed_len()` |
| tree insert | `radix_cache.py` `insert` | `len(key)`, `prefix_len` |

3. **요청 딱 1개**를 보낸다 (`max_new_tokens=8`).
4. 로그를 시간순으로 정렬해 **콜 체인 다이어그램**을 그린다.
   Part 2 §10.1과 같은 형식으로.

### 답해야 할 질문

1. KV 슬롯은 총 몇 번 할당되는가? 프리필에서 1번 + 디코드마다 1번씩 = 몇 번인가?
2. **KV가 정확히 어디서 해제되는가?** `release_kv_cache` 안에서 `free()`가 몇 번 불리는가?
   각각 무엇을 해제하는가? (힌트: 중복 인덱스와 정렬 안 된 tail, 두 종류다)
3. `dec_lock_ref`는 언제 불리는가? 그 시점에 `evictable_size_`는 얼마나 증가하는가?
4. 같은 요청을 **두 번째로** 보내면 (캐시 히트) 위 체인의 어느 단계가 사라지는가?
   `extend_input_len`은 얼마가 되는가?
5. 요청 종료 후 그 KV는 **즉시 free되는가, tree에 남는가?**
   Session 2의 `cache_finished_req`를 근거로 답하라.

### 제출물

- 삽입한 로그 패치 (diff 형식)
- 시간순 로그 전문 (요청 1개분)
- 콜 체인 다이어그램 (Part 2 §10.1 형식)
- 질문 1~5에 대한 답

---

## 제출 템플릿 (`report.md`)

```markdown
# Session 3 실습 리포트
- 이름:
- 브랜치 / 커밋 해시:
- GPU / 모델:
- 서버 KV 풀 크기 (get_server_info 기준):

## 실습 결과 요약
| 실습 | 핵심 관찰 | 예상과 달랐던 점 |
|---|---|---|
| 1. 어드미션 | | |
| 2. LPM vs FCFS | | |
| 3. 리트랙션 | | |
| 4. 청크 프리필 | | |

## 과제 1 — 손으로 어드미션 추적
(위 형식대로)

## 과제 2 — batch_is_full 스톨
...

## 배운 것 / 아직 모르겠는 것
- 여전히 헷갈리는 지점 3가지를 반드시 쓸 것. 다음 세션에서 다룬다.
```

---

## 채점 기준

| 항목 | 배점 | 기준 |
|---|---|---|
| 실습 1~3 수행 및 데이터 수집 | 30 | 로그 파일 + 집계표 제출 |
| 과제 1 (손 추적) | 15 | 계산 과정이 재현 가능한가 |
| 과제 2 (스톨 재현) | 15 | 실제로 재현했는가, 원인을 코드로 지목했는가 |
| 과제 3~4 (소스 검증) | 20 | 문서와 코드의 차이를 스스로 찾았는가 |
| 과제 5~6 (심화) | 20 | 둘 중 **하나 선택** |
| **가산점** | +10 | 문서(Part 1/2)의 오류를 발견해 근거와 함께 지적 |

> 마지막 항목이 진심이다. 이 문서들의 라인 번호와 조건식은 브랜치 기준으로 작성되었고,
> 코드는 계속 바뀐다. **틀린 곳을 찾아오는 것이 가장 좋은 학습 증거다.**
