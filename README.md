# BDAIFin — Real-Time FDS (2-Stage Fraud Detection System)

> **온라인 카드 거래 사기 탐지를 위한 2-Stage 의사결정 파이프라인**
> Latency vs Context의 본질적 충돌을 **구조 분리**로 해결한 실시간 FDS 설계 프로젝트.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-LR-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Stage2-green.svg)](https://lightgbm.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

**BDAI FinDA 금융데이터분석 심화과정 1기 · 우수 수료**

---

## 1. Why 2-Stage?

온라인 카드 거래 사기 탐지는 **두 개의 상반된 제약**과 동시에 싸워야 합니다.

| 제약 | 요구 |
|---|---|
| **Latency** | 실시간 결제 환경 — 수십~수백 ms 안에 판단 필요 |
| **Context** | 사기 탐지는 단일 거래만으로 불가능 — 사용자·카드·히스토리 맥락 필수 |

**Latency vs Context** 는 한 모델에서 동시에 만족할 수 없는 trade-off.
이 프로젝트는 **레이어를 분리해 두 제약을 각각 해결**합니다.

```
┌─────────────────────────────────────────────────────────┐
│         Stage 1: Transaction-Only (Latency 우선)       │
│         거래 단독 정보 → 빠른 스코어링 → Review 선별    │
└─────────────────────────────────────────────────────────┘
                          ↓
             (Stage1이 선별한 의심 거래만)
                          ↓
┌─────────────────────────────────────────────────────────┐
│         Stage 2: Context & History (Precision 우선)     │
│         사용자·카드·히스토리 결합 → 정밀 판별           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│         CHECK: Drift Validation                         │
│         운영 환경 분포 이동 감지 및 대응 전략 검증      │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Key Results

| Layer | Model | Recall | Precision | F1 | Inference Time |
|---|---|---|---|---|---|
| **Stage 1** | Logistic Regression | 0.455 | 0.834 | 0.589 | **0.037s** |
| **Stage 2** | LightGBM | 0.955 | 0.987 | **0.971** | ~1ms |

**모델 선택 기준:**

- **Stage 1**: Recall 최상위 + 가장 빠른 추론 시간 → 거래 단독 피처의 선형 분리 가능성이 높고 연산이 가벼움
- **Stage 2**: 히스토리 피처의 복잡한 비선형 상호작용 포착 필요 → 트리 기반 필수, lat_p50 0.96ms로 운영 효율 균형

---

## 3. Repository Structure

```
BDAIFin/
├── EDA&FEATURE/              # 분석 & 피처 의사결정
│   ├── EDA_CARD.ipynb        # 카드 속성 리스크 분석
│   ├── EDA_ERR.ipynb         # 오류 패턴 (CVV/PIN) 리스크
│   ├── EDA_HISTORY.ipynb     # 카드/클라이언트 히스토리
│   ├── EDA_INTERACTION.ipynb # 상호작용 피처 구조
│   ├── EDA_MCC.ipynb         # MCC 리스크 (Bayesian smoothing)
│   ├── EDA_MONEY.ipynb       # 금액/소득/한도 비율
│   ├── EDA_PERSON.ipynb      # 개인 속성 분석
│   ├── EDA_TIME.ipynb        # 시간 기반 (cyclic encoding)
│   └── FEATURESELECT.ipynb   # 최종 피처 셋 확정
│
└── PIPELINE/
    ├── STAGE1/               # Transaction-Only Layer
    │   ├── DATA1.py          # 거래 정제 & 기본 파생 피처
    │   ├── DATA2.py          # Stage1 artifacts & 데이터셋
    │   └── MODEL.ipynb       # LR 최적 설정 + threshold
    │
    ├── STAGE2/               # Context & History Layer
    │   ├── DATA0.py          # 베이스 테이블 (거래+사용자+카드)
    │   ├── DATA1.py          # 히스토리 기반 피처 (TRAIN)
    │   ├── DATA2.py          # Review ids 기반 TEST 생성
    │   └── MODEL.ipynb       # LightGBM + threshold 확정
    │
    └── CHECK/                # Drift Validation
        └── ...               # 운영 환경 분포 이동 대응
```

---

## 4. Stage 1 — Transaction-Only Layer

### Purpose

- 전체 거래를 **빠르게 스코어링**
- Stage 2로 넘길 **Review 대상 선별**
- 최소 정보 기반의 안정적 1차 필터

### Key Artifacts

Stage 1은 **재학습 없이 artifacts 업데이트만으로 운영 가능**한 구조로 설계.

- `mcc_smoothed_risk_map` — MCC별 Bayesian smoothing 적용 위험도
- `mcc_risk_level` — MCC 위험 등급화
- `risky_wd_hour` — 요일×시간대 위험 조합
- `high_risk_months` — 시간대 편향 분석 결과

### Final Features

```
- log_abs_amount         # 금액 로그 변환 (heavy-tail 완화)
- hour_sin, hour_cos     # 시간 주기성 (cyclic encoding)
- month_sin              # 월 주기성
- mcc_smoothed_risk      # MCC Bayesian smoothing 위험도
- mcc_risk_level         # MCC 위험 등급
- is_risky_wd_hour       # 위험 시간대 flag
- err_bad_cvv            # CVV 오류 flag
```

### Key Design Decision: Cyclic Encoding

raw hour(0~23)를 그대로 쓰면 모델은 23시와 0시를 가장 먼 값으로 인식해 시간의 **주기성** 을 학습할 수 없음.
`sin/cos` 변환으로 시간 주기를 보존 → **raw hour 대비 +29% 성능 개선**.

---

## 5. Stage 2 — Context & History Layer

### Purpose

- Stage 1이 선별한 거래에 대해 **정밀 판별**
- 행동 패턴 기반 이상 탐지
- **속도·신규성·변화** 기반 리스크 강화

### Base Table Construction

거래 + 사용자 + 카드 데이터 결합:

- **계정/만료 파생**: `months_to_expire`, `months_from_account`, `years_since_pin_change`, `years_to_retirement`
- **카드 속성**: `is_credit`, `is_prepaid`, `has_chip`, `cb_Visa/Mastercard/Amex/Discover`
- **금액/소득/한도 비율**: `amount_income_ratio`, `amount_limit_ratio` + 로그 변환

### History Features

```
- velocity_spike_ratio      # 속도 급증 (단기 거래 폭증)
- client_mcc_is_new         # 클라이언트-MCC 신규 조합
- merchant_is_new           # 신규 가맹점
- merchant_change_last5     # 가맹점 변화 빈도
- log_interval_dev          # 거래 간격 이상
- limit_ratio_extreme       # 극단 한도 사용
```

### Critical Design: Temporal Leakage 방지

> **히스토리 계산은 전체 raw 데이터를 기준으로 수행하고, 마지막 단계에서 Review ids만 필터링**

이 순서를 지키지 않으면 미래 정보가 과거로 유입되어 운영 환경에서 성능이 무너짐.
시계열 데이터에서 가장 흔하고 치명적인 실수를 구조적으로 차단한 설계.

---

## 6. Feature Engineering Highlights

### MCC Bayesian Smoothing

**문제:** raw fraud rate를 그대로 쓰면 거래량이 적은 MCC에서 rate=1.0인 노이즈가 다수 발생.

**해결 과정:**

1. 거래량 vs raw fraud rate 산점도로 **노이즈 시각적 입증**
2. Bayesian smoothing 적용 (α = 100/500/1000/2000 민감도 분석)
3. 연도별 Jaccard Overlap 0.6~0.85로 **구조적 안정성 검증**
4. **α=1000 최종 채택**

**결과:** Baseline 대비 ΔPR +0.553, ΔLift +4.49

### Extreme Value Handling

**가설 (직관):** outlier clipping이 정답일 것

**검증 결과:** 상위 0.1% 구간의 사기율이 **66.5%** → 극단값 자체가 핵심 사기 신호

**결정:** clipping 대신 원본 ratio 유지 + `limit_ratio_extreme` flag 추가 → Top-decile Lift +0.329

> **교훈:** 가설은 빠르게 세우되, 데이터가 반박하면 즉시 폐기한다.

---

## 7. Feature Validation Workflow

모든 피처는 다음 4단계 검증을 통과해야 모델에 투입:

```
1. EDA
   - 데이터 구조·분포의 특이점 식별

2. 다중공선성 검증 (VIF, Spearman)
   - 중복 정보 제거

3. SHAP 기반 비선형 기여도 분석
   - 실제 모델 성능 기여도 정량화

4. Ablation
   - 피처 제거 시 성능 변화로 효용 입증
```

이후 **Covariate Shift (PSI · KS)** 분석으로 시간에 따른 drift까지 점검.

---

## 8. CHECK — Drift Validation

운영 환경에서의 분포 이동 대응을 검증하는 영역.

### Purpose

- 데이터 분포 변화 감지
- artifacts 업데이트 여부 판단
- threshold 재조정 필요성 검증

### Key Finding

- `mcc_smoothed_risk` 에서 PSI 1.805 (강한 drift) 감지
- **모델 재학습 없이 artifacts 업데이트만으로 대응 가능한 구조**임을 검증
- → 운영 비용 최소화 + 모델 안정성 동시 확보

---

## 9. End-to-End Logic

```
1. EDA에서 피처 후보 발굴
2. Feature Selection으로 최종 피처 확정
3. Stage1 데이터 생성 (artifacts 기반)
4. Stage1 모델 학습 및 Review ids 선별
5. Stage2 데이터 생성 (Temporal Leakage 방지)
6. Stage2 모델 학습 및 최종 판별
7. Drift 감지 및 artifacts 업데이트 전략 검증
```

---

## 10. Design Philosophy

- 2-Stage 구조는 **속도와 정밀도의 분리 설계**
- Stage 1은 거래 단독 기반의 빠른 필터
- Stage 2는 컨텍스트 + 행동 기반의 심층 분석
- 모든 히스토리 피처는 **시간 정렬 후 생성** (leakage 방지)
- 운영 안정성을 고려한 **artifacts 기반 설계** (재학습 최소화)
- Drift 대응을 고려한 **구조적 확장 가능성** 확보

---

## 11. What I Learned

이 프로젝트의 가장 큰 수확은 **"모델보다 평가 설계가 운영 의사결정을 좌우한다"** 는 통찰이었습니다.

- PR-AUC와 Precision@Recall로 6가지 샘플링·class weight 조합을 비교하며 평가 지표 선택이 모델의 운영 방향을 바꾼다는 것을 체득
- 7종 파생 피처 각각을 EDA → 다중공선성 → SHAP → Ablation의 4단계 검증으로 통과시키며 **데이터 품질 관리는 꼼꼼한 반복 작업의 인내** 라는 것을 배움
- Latency vs Context 라는 단일 제약을 출발점으로 모든 의사결정을 일관되게 연결하며 **분석 흐름의 논리성** 이 결과보다 중요하다는 것을 깨달음

---

## 12. Team & Role

**Team 4 (4인 팀, 팀장)** — 김나경 · 김채현 · 김형준 · 정준우

기여:

- 전체 파이프라인 구조 설계 (Latency vs Context trade-off 정의)
- Stage 1/2 피처 의사결정 주도 (MCC smoothing, extreme value, cyclic encoding)
- 4단계 검증 워크플로우 수립 및 팀 내 적용
- Tableau 모니터링 대시보드 구축
- **우수 수료자 선정**

---

## 13. Contact

- GitHub: [carolyn0515](https://github.com/carolyn0515)
- Email: neige040515@gmail.com
