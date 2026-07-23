# Invest-Signal

4시간봉 마감마다 **바이낸스 전체 USDT 무기한 + 레버리지 ETF**를 스캔해서
시그널이 뜨면 **텔레그램으로 알림**을 보내는 스캐너.

## 시그널 (모두 4h봉 기준)

### 상승초입 (uptrend_onset)

1. **역배열** — `MA120 < MA240 < MA480`
2. **240선 터치** — 캔들 고가가 240선에 닿음 (윗꼬리 터치 인정)
3. **60선 이탈** — 터치 후 `touch_window_bars`(기본 60봉 = 10일) 이내에
   종가가 60선 아래로 마감하는 **첫 봉**
4. **분기 VWAP 필터** — 트리거 봉 종가가 분기 앵커드 VWAP(QVWAP) 위여야
   함 (`qvwap_condition: false`로 끌 수 있음)

같은 셋업으로는 한 번만 알리고, 240선을 다시 터치하면 새 셋업으로 다시
알릴 수 있다.

### 풀백 (pullback)

1. **풀백 구간 진입** — 240선이 480선 **위로 올라온 상태** (골든크로스
   이후). 이 구간에서는 상승초입이 발동하지 않는다 (두 시그널은 240/480
   위치로 국면을 나눈다: 240<480 = 상승초입 구간, 240>480 = 풀백 구간)
2. **120선 하회** — 종가가 120선 아래로 마감하는 순간 알림
3. **크립토 랭크 필터** — 크립토는 주도주만: 24h 거래대금 상위
   `volume_top`(기본 100) 또는 최근 7일 상승률 상위 `gain_top`(기본 50)에
   들고, 24h 거래대금이 `min_turnover_usd`(기본 $5M) 이상이어야 알림.
   ETF·주식·MSS에는 적용하지 않는다

연속 하회 구간은 첫 봉만 알리고, 120선 위로 복귀 후 다시 하회하면 새
풀백으로 다시 알린다.

### 하락전환 (downtrend_reversal) — bearish CHoCH

1. 종가가 **60선 아래로 눌렸던 구간**이 완성되고 (종가가 60선 위로 복귀)
2. 그 구간의 **최저 저가(꼬리 포함) = 직전저점**을 기준으로
3. 이후 `lookback_bars`(기본 180봉 = 30일) 안에 종가가 직전저점 아래로
   **처음 마감**하는 순간 알림

진행 중인 눌림 안에서 저점을 갱신하는 것은 돌파로 치지 않는다 — 완성된
직전저점의 하향돌파(ICT의 CHoCH)만 잡는다.

### MSS (mss)

조건 없이 **직전 스윙저점**(좌우 `pivot_k`=6봉(24시간)보다 낮은 저가로
확정된 저점)을 종가가 하향돌파하는 순간 알림. 하락전환보다 넓게 잡히는 구조
돌파 신호로, 첫 돌파 봉만 알리고 저점 위로 복귀 후 재돌파하면 다시 알린다.

### 알림 메시지 구성

- **신규 시그널**: 시그널별(🟢상승초입 🔵풀백 — MSS는 풀백 칸에 태그) → 그 아래
  [크립토]/[ETF]/[주식] 그룹. 크립토는 USDT 접미사 없이 표기, 차트 링크·QVWAP·배열 상태 포함
- **↳ 추적 리스트**: 각 시그널·시장 칸 아래 한 줄 — `심볼(경과일d·배열)`
  - 상승초입: 발생 후 계속 추적(60선 위 복귀해도 유지) — 종가가 240선 위로
    올라섰다가 **다시 240선 아래로 마감하면(돌파 실패) 제거**. CHoCH로는
    제거하지 않는다
  - 눌림목: 발생 후 계속 표시 — 120선 위로 복귀해 올라가는 동안도 추적하며,
    **하락전환(CHoCH)이 나와야 제거**된다
  - MSS: 종가가 계속 깨진 직전저점 아래인 동안 (경고 마커로 병행 표시)
  - 최근 7일(42봉) 내 트리거만 조회, 시그널당 최신 15종목까지 표시
- **하락전환은 표시하지 않음** — 셋업 이후 하락전환이 발생한 종목의
  눌림목 항목을 유지 중 리스트에서 영구 제거하는 역할만 한다

새 시그널은 `invest_signal/signals/`에 모듈을 추가하고
`config.py`의 `detectors()`에 등록하면 된다.

## 감시 대상

- **크립토**: 바이낸스에서 거래 중인 전체 USDT 무기한 (자동 갱신)
- **레버리지 ETF**: `config.yaml`의 18종 — 미국 섹터·지수 불/베어
  (SOXL/SOXS, TQQQ/SQQQ, FAS, LABU, ERX, NUGT, YINN, RETL, TMF, DFEN,
  BITU, ETHU) + 한국 (KODEX 레버리지/코스닥150레버리지 및 인버스2X 쌍)
- **주식 (매일 자동 갱신)**: Quant-Portfolio 레포의 `reports/latest.md`에서
  매 스캔마다 로드 — 미국·한국 각각 강한 섹터 대장주(2×3)와 개별 RS Top5,
  약 20종. 코스피/코스닥은 `.KS` 실패 시 `.KQ`로 자동 재시도.
  - Quant-Portfolio가 **퍼블릭이면 추가 설정 불필요**. 프라이빗이면
    `QP_GITHUB_TOKEN` 시크릿 필요 — Contents Read-only fine-grained PAT
    ([github.com/settings/personal-access-tokens](https://github.com/settings/personal-access-tokens)
    → Only select repositories → Quant-Portfolio → Contents: Read-only)
  - 조회 실패 시 `config.yaml`의 정적 스냅샷으로 폴백

## 바이낸스 지역 차단(451)과 데이터 소스

GitHub Actions 러너(미국 IP) 실측 결과:

| 엔드포인트 | 결과 |
|---|---|
| `fapi.binance.com` (선물) | ❌ HTTP 451 |
| `api.binance.com` (현물) | ❌ HTTP 451 |
| `data-api.binance.vision` (공식 현물 데이터 미러) | ✅ 200 |

`crypto.source: auto`(기본)는 **fapi 시도 → 실패하면 현물 미러로 자동
폴백**한다. 현물·무기한의 4h 가격은 MA 시그널 용도로는 사실상 동일하다.
단, 커버리지 차이가 있다(실측): 활성 USDT 퍼프 787종 중 현물로 커버되는
건 377종(48%) — 나머지는 토큰화 주식 퍼프, 퍼프 전용 밈코인, 현물 상폐
코인 등. BTC·ETH·SOL 등 주요 코인은 전부 커버된다. 진짜 무기한 데이터로
돌리려면:

- `BINANCE_FAPI_BASE` 시크릿에 비미국 프록시 주소를 넣거나
- 한국 PC/서버에서 crontab으로 실행 (`--only crypto`)

## 설정

### 1. 텔레그램 봇 (필수)

1. [@BotFather](https://t.me/BotFather)에서 봇 생성 → 토큰 복사
2. 봇에게 아무 메시지 전송 후 `https://api.telegram.org/bot<토큰>/getUpdates`에서
   `chat.id` 확인
3. 레포 Settings → Secrets and variables → Actions에 등록:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

시크릿이 비어 있으면 스캔 워크플로가 **즉시 실패**한다(빨간 런) — 알림이
조용히 증발하는 것을 막기 위한 안전장치다. 발송에 실패한 시그널은 상태에
기록하지 않으므로 다음 4h 스캔에서 한 번 더 시도된다(`grace_bars`).

### 2. 스케줄

`.github/workflows/scan.yml`이 KST 07/11/15/19/23시(하루 5회, 심야 제외)에
자동 실행된다. (GitHub 크론 특성상 수십 분 지연될 수 있음)
`scan.yml`을 수정해 푸시하면 즉시 1회 실행된다.
**schedule 트리거는 기본 브랜치에서만 동작**하므로 이 브랜치를 main에
머지해야 활성화된다. 중복 알림은 `state/alerts_state.json`으로 방지하며
실행 후 자동 커밋된다.

### 3. 로컬 실행

```bash
pip install -r requirements.txt
python -m invest_signal --dry-run          # 발송 없이 결과만 출력
python -m invest_signal --only crypto      # 크립토만
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python -m invest_signal
```

### 4. config.yaml

```yaml
signal:
  uptrend_onset:
    enabled: true
    touch_window_bars: 60      # 터치 유효기간(봉) — 60봉 = 10일
    grace_bars: 1              # 스캔 한 번 놓쳤을 때 소급 허용 봉 수
    qvwap_condition: true      # 트리거 종가 > 분기 VWAP 요구
  pullback:
    enabled: true
    ma_entry: 120              # 하회 판정 기준선
    grace_bars: 1
crypto:
  source: auto                 # auto | fapi | spot_mirror
  exclude: []                  # 제외 심볼
etf:
  tickers: [...]               # 코드/시장/이름
```

의존성은 `requirements.txt`에 정확히 고정돼 있다(무인 크론이라 상위
버전이 갑자기 깨는 것을 방지). 올릴 때는 테스트 후 갱신할 것.

## 테스트

```bash
pip install -r requirements-dev.txt
pytest
```

푸시하면 `Test & Dry-Run` 워크플로가 pytest + 실데이터 dry-run으로 검증한다.
