# MCP Advanced Data Analysis System

**26개의 전문가급 데이터 분석 도구**를 제공하는 MCP(Model Context Protocol) 기반 데이터 분석 시스템

## ✨ 주요 기능

### 📊 데이터 탐색 (4개 도구)
- `get_dataset_info` - 기본 정보 확인
- `profile_dataset` - 종합 프로파일링
- `detect_data_types` - 자동 타입 분류
- `find_duplicates` - 중복 탐지

### 🧹 데이터 전처리 (5개 도구)
- `handle_missing_values` - 결측치 처리
- `detect_outliers`, `remove_outliers` - 이상치 탐지/제거
- `encode_categorical` - 범주형 인코딩 (Label/One-hot)
- `scale_features` - 특성 스케일링 (Standard/MinMax)

### 📈 시각화 (7개 도구)
- `plot_histogram` - 히스토그램 (커스터마이징 가능)
- `plot_boxplot` - 박스플롯
- `plot_scatter` - 산점도 (레전드, 색상 등 완전 커스텀)
- `plot_correlation_heatmap` - 상관관계 히트맵
- `calculate_correlation` - 상관계수 계산
- `analyze_target_distribution` - 타겟 분포 분석

### 🤖 머신러닝 (2개 도구)
- `compare_models` - RandomForest, XGBoost, LogisticRegression 비교
- `evaluate_model` - Confusion Matrix, Feature Importance

### 📐 통계 분석 (5개 도구)
- `test_normality` - Shapiro-Wilk 정규성 검정
- `test_ttest` - 독립 T-검정
- `test_anova` - 일원 분산분석
- `test_chi_square` - 카이제곱 독립성 검정
- `calculate_confidence_interval` - 신뢰구간 계산

### 💾 데이터 관리 (3개 도구)
- `list_cached_datasets` - 캐시 모니터링
- `clear_cache` - 캐시 초기화
- Smart Caching System - 자동 메모리 최적화

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost
pip install mcp langchain-mcp-adapters langgraph langchain-openai
pip install langchain-ollama  # Ollama 사용 시
```

### 2. 서버 실행

```bash
python data_server_v.3.0.py
```

### 3. 클라이언트 실행 (별도 터미널)

```bash
python data_client.py
```

## 💡 사용 예시

### 데이터 탐색
```
You: customer_churn.csv를 프로파일링해줘

AI: [통계량, 결측치, 상관관계 등 종합 분석]
```

### 시각화
```
You: area와 price의 산점도를 그려줘. bedrooms로 색상 구분하고, 
     레전드 제목은 '방 개수', 위치는 우측 상단으로 해줘

AI: [커스터마이징된 산점도 생성]
```

### 머신러닝
```
You: customer_churn.csv에서 churn을 예측하는 모델을 비교하고 평가해줘

AI: [RandomForest, XGBoost, LR 성능 비교 → 최고 모델 상세 평가]
```

## 🎨 시각화 커스터마이징

### plot_scatter 고급 옵션
```python
plot_scatter(
    csv_path="house_price.csv",
    x_column="area",
    y_column="price",
    hue_column="bedrooms",
    title="주택 면적과 가격 관계",
    xlabel="면적 (sqft)",
    ylabel="가격 ($)",
    figsize_width=12,
    figsize_height=8,
    marker_size=80,
    alpha=0.7,
    color_palette="Set2",
    show_legend=True,
    legend_title="방 개수",
    legend_position="upper left"
)
```

## 🔧 모델 설정

### OpenAI 사용
```python
# data_client.py
USE_OLLAMA = False
MODEL_NAME = "gpt-4o-mini"
```

### Ollama 사용 (무료)
```python
# data_client.py
USE_OLLAMA = True
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "qwen2.5:72b"
```

## 📦 테스트 데이터셋

- `customer_churn.csv` - 7,043행, 분류 (이탈 예측)
- `house_price.csv` - 545행, 회귀 (가격 예측)
- `sales_timeseries.csv` - 1,000일, 시계열 분석

생성:
```bash
python generate_all_test_data.py
```

## 📚 문서

- [사용 가이드](usage_guide.md) - 20+ 분석 시나리오
- [원격 Ollama 설정](remote_ollama_setup.md) - SSH 서버 연동

## 🎯 주요 특징

- ✅ **스마트 캐싱** - 반복 작업 50% 속도 향상
- ✅ **대화 히스토리** - 연속적인 자연어 대화
- ✅ **한글 지원** - 완전한 한국어 인터페이스
- ✅ **즉시 실행** - AI가 바로 도구 호출
- ✅ **커스터마이징** - 시각화 상세 조절
- ✅ **무료 옵션** - Ollama 지원

## 📊 시스템 요구사항

- Python 3.11+
- 8GB+ RAM (CPU 모드)
- 또는 GPU (VRAM 18GB+, Ollama 사용 시)

## 🤝 기여

이슈 및 PR 환영합니다!

## 📄 라이센스

MIT License

