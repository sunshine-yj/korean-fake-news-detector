# 데이터베이스

MySQL 8.0 기반. 수집한 실제 기사와 생성한 가짜 기사를 단일 `articles` 테이블에 `is_fake` 라벨로 구분해 저장한다.

```
database/
├── schema.sql          최종 정리 스키마 (신규 구축용)
├── migrations/         변경 이력
│   ├── 001_add_prompt_type.sql
│   ├── 002_add_clean_columns.sql
│   ├── 003_add_missing_sources.sql
│   └── 004_add_origin_reference.sql
└── queries/
    └── stats.sql       통계 및 검증 쿼리
```

---

## 1. 구축 방법

### 신규 구축

```bash
mysql -u root -p < schema.sql
```

`schema.sql`은 마이그레이션이 모두 반영된 최종 형태다. 새로 시작하는 경우 이 파일 하나면 된다.

### 기존 DB 갱신

초기 DDL로 구축된 DB가 있다면 마이그레이션을 순서대로 적용한다.

```bash
mysql -u root -p < migrations/001_add_prompt_type.sql
mysql -u root -p < migrations/002_add_clean_columns.sql
mysql -u root -p < migrations/003_add_missing_sources.sql
mysql -u root -p < migrations/004_add_origin_reference.sql
```

---

## 2. 테이블 구조

### articles

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `article_id` | BIGINT PK | 기사 고유 식별자 |
| `title` | VARCHAR(500) | 기사 제목 |
| `content` | TEXT | 기사 본문 (21,000자 절단) |
| `title_clean` | VARCHAR(500) | 전처리된 제목 (모델 입력) |
| `content_clean` | TEXT | 전처리된 본문 (모델 입력) |
| `url` | VARCHAR(1000) | 원문 URL |
| `published_at` | DATETIME | 발행 일시 |
| `is_fake` | TINYINT | 0=진짜, 1=가짜 |
| `prompt_type` | TINYINT | 조작 유형 1~5 (진짜는 NULL) |
| `origin_article_id` | BIGINT FK | 가짜 기사의 원본 참조 |
| `source_id` | INT FK | 언론사 |
| `category_id` | INT FK | 카테고리 |
| `created_at` | TIMESTAMP | 레코드 생성 시각 |

### news_sources

| source_id | 언론사 | 도메인 |
|---:|---|---|
| 1 | 연합뉴스 | yna.co.kr |
| 2 | 노컷뉴스 | nocutnews.co.kr |
| 3 | MBC | imbc.com |
| 4 | SBS | sbs.co.kr |
| 5 | 조선일보 | chosun.com |
| 6 | 한겨레 | hani.co.kr |
| 7 | 동아일보 | donga.com |
| 8 | 네이버뉴스 | news.naver.com |
| 9 | GPT-4o | — |
| 10 | JTBC | jtbc.co.kr |
| 11 | GPT-5-mini | — |

실제 수집 대상은 연합뉴스·SBS·동아일보·한겨레·JTBC 5개사다. 나머지는 초기 설계 단계에서 등록된 후 사용되지 않았다.

### categories

| category_id | 이름 |
|---:|---|
| 1 | 정치 |
| 2 | 경제 |
| 3 | 사회 |
| 4 | 문화 |
| 5 | 국제 |

---

## 3. 변경 이력

프로젝트가 진행되며 스키마가 확장되었다. 각 마이그레이션의 도입 배경은 다음과 같다.

| # | 내용 | 배경 |
|---|---|---|
| 001 | `prompt_type` 추가 | 가짜뉴스 생성 파이프라인 도입 시 조작 유형 기록 필요 |
| 002 | `title_clean`, `content_clean` 추가 | 원문 보존과 모델 입력 분리 |
| 003 | 동아일보·JTBC 시드 추가 | 초기 시드 누락으로 FK 제약 위반 발생 |
| 004 | `origin_article_id` 추가 | 원본-가짜 쌍 추적 |

**004는 데이터 구축 완료 이후 정의한 컬럼이다.** 기존 레코드는 값이 NULL이며, `url` 매칭을 통한 백필이 가능하다(마이그레이션 파일 하단 주석 참조).

---

## 4. 알려진 문제

데이터 해석 시 반드시 고려해야 할 사항이다. 상세 배경은 [`../pipeline/README.md`](../pipeline/README.md) 7절 참조.

### 4.1 source_id 매핑 불일치 — 영향 큼

두 수집 워크플로가 서로 다른 `source_id` 값을 하드코딩했다.

| 언론사 | RSS 수집 | 네이버 API 수집 | 시드 정의 |
|---|---:|---:|---:|
| 연합뉴스 | 1 | 1 | 1 |
| SBS | 4 | **6** | 4 |
| 동아일보 | 7 | 7 | 7 |
| 한겨레 | **6** | **8** | 6 |
| JTBC | **11** | **10** | 10 |

`source_id = 6`이 수집 경로에 따라 한겨레 또는 SBS를 가리키므로, **단일 값에 두 언론사가 혼재**한다. 또한 `source_id = 8`(네이버뉴스)에 한겨레 실기사가, `11`(GPT-5-mini)에 JTBC 실기사가 저장되었을 수 있다.

**언론사별 집계는 `source_id`가 아닌 URL 도메인 기준으로 수행할 것.** `queries/stats.sql` 2-1 참조.

### 4.2 카테고리 정보 소실

RSS 수집 워크플로는 정치·경제·사회 피드를 각각 요청하지만, `<item>` 블록 병합 과정에서 출처 섹션 정보가 사라진다. 이 경로로 수집된 기사의 `category_id`는 모두 동일한 값으로 저장된다.

네이버 API 수집은 키워드 카테고리를 유지하므로 이 문제가 없다.

실제 분포는 `queries/stats.sql` 3번으로 확인한다.

### 4.3 URL 중복 제약 부재

`url`에 UNIQUE 제약이 없다. 중복 방지를 n8n 워크플로의 `SELECT COUNT(*)` + IF 노드로 처리했으므로, 동시 실행 시 중복 삽입이 가능하다.

제약을 추가하려면 가짜 기사가 원본 URL을 복사하는 구조를 먼저 해결해야 한다(가짜의 `url`을 NULL로 두거나 `origin_url` 컬럼으로 분리).

### 4.4 가짜 기사의 URL

가짜 기사 레코드가 원본 기사의 `url`을 그대로 보유한다. 해당 URL의 실제 기사와 저장된 내용이 다르므로, 데이터셋 공개 시 이 필드를 제외하거나 별도 컬럼으로 분리하는 것이 바람직하다.

동시에 이 구조가 `origin_article_id` 부재 상황에서 원본-가짜를 연결하는 유일한 수단이기도 하다.

### 4.5 발행일 폴백

RSS `pubDate`가 없는 경우 수집 시각을 `published_at`으로 저장한다. `published_at`과 `created_at`의 차이가 5분 미만인 레코드가 이에 해당할 가능성이 높다.

### 4.6 본문 길이 절단

MySQL TEXT 용량(65,535 bytes)과 한글 3바이트 인코딩을 고려해 본문을 21,000자에서 절단한다. 긴 기사는 후반부가 손실되었다.

---

## 5. 검증 절차

DB 복구 후 다음 순서로 실행하여 데이터셋 명세서의 수치를 확정한다.

```bash
mysql -u root -p fake_news_db < queries/stats.sql
```

| 쿼리 | 확인 항목 | 반영 위치 |
|---|---|---|
| 1-1, 1-2 | 진짜/가짜 비율, 유형별 건수 | `data/README.md` 3절 |
| 2-1 | 언론사 분포 (URL 기준) | `data/stats/source_distribution.csv` |
| 2-3 | source_id 혼재 여부 | 본 문서 4.1 확정 |
| 3 | 카테고리 분포 | `data/README.md` 3.4절 |
| 4-1, 4-2 | 원본 중복 생성 여부 | `pipeline/README.md` 한계 ⑥ |
| 5-2 | JTBC 수집 건수 | `pipeline/README.md` 한계 ③ |
| 5-4, 5-5 | 발행일 범위 | `data/README.md` 1절 |
| 6 | 본문 길이 분포 | `data/stats/length_distribution.csv` |
| 7-1, 7-2 | 생성 품질 | `data/README.md` 8절 |
