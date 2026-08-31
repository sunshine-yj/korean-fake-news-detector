-- ================================================
-- 데이터셋 통계 및 검증 쿼리
--
-- DB 복구 후 실행하여 data/README.md 및 data/stats/ 의
-- 수치를 확정한다.
-- ================================================

USE fake_news_db;


-- ------------------------------------------------
-- 1. 기본 분포
-- ------------------------------------------------

-- 1-1. 진짜/가짜 비율
SELECT is_fake, COUNT(*) AS cnt
FROM articles
GROUP BY is_fake;

-- 1-2. 조작 유형별 건수
SELECT
    prompt_type,
    CASE prompt_type
        WHEN 1 THEN '수치 변경'
        WHEN 2 THEN '고유명사 치환'
        WHEN 3 THEN '출처 익명화'
        WHEN 4 THEN '인과 왜곡'
        WHEN 5 THEN '동사 반전'
    END AS type_name,
    COUNT(*) AS cnt
FROM articles
WHERE is_fake = 1
GROUP BY prompt_type
ORDER BY prompt_type;


-- ------------------------------------------------
-- 2. 언론사 분포
--
-- source_id 는 두 수집 파이프라인 간 매핑이 불일치하므로
-- (README.md 참조) URL 도메인 기준 집계를 신뢰한다.
-- ------------------------------------------------

-- 2-1. URL 도메인 기준 (권장)
SELECT
    CASE
        WHEN url LIKE '%yna.co.kr%'  THEN '연합뉴스'
        WHEN url LIKE '%sbs.co.kr%'  THEN 'SBS'
        WHEN url LIKE '%donga.com%'  THEN '동아일보'
        WHEN url LIKE '%hani.co.kr%' THEN '한겨레'
        WHEN url LIKE '%jtbc.co.kr%' THEN 'JTBC'
        ELSE '기타'
    END AS source_name,
    COUNT(*) AS cnt
FROM articles
WHERE is_fake = 0
GROUP BY source_name
ORDER BY cnt DESC;

-- 2-2. source_id 기준 (불일치 확인용)
SELECT
    a.source_id,
    s.source_name AS registered_name,
    COUNT(*) AS cnt
FROM articles a
LEFT JOIN news_sources s ON a.source_id = s.source_id
WHERE a.is_fake = 0
GROUP BY a.source_id, s.source_name
ORDER BY a.source_id;

-- 2-3. source_id 와 실제 도메인의 교차 검증
--      한 source_id 에 여러 언론사가 섞여 있는지 확인
SELECT
    a.source_id,
    s.source_name AS registered_name,
    CASE
        WHEN a.url LIKE '%yna.co.kr%'  THEN '연합뉴스'
        WHEN a.url LIKE '%sbs.co.kr%'  THEN 'SBS'
        WHEN a.url LIKE '%donga.com%'  THEN '동아일보'
        WHEN a.url LIKE '%hani.co.kr%' THEN '한겨레'
        WHEN a.url LIKE '%jtbc.co.kr%' THEN 'JTBC'
        ELSE '기타'
    END AS actual_source,
    COUNT(*) AS cnt
FROM articles a
LEFT JOIN news_sources s ON a.source_id = s.source_id
WHERE a.is_fake = 0
GROUP BY 1, 2, 3
ORDER BY a.source_id, cnt DESC;


-- ------------------------------------------------
-- 3. 카테고리 분포
--
-- RSS 수집 경로는 병합 과정에서 섹션 정보가 소실되어
-- category_id 가 단일 값으로 고정되었을 가능성이 있다.
-- ------------------------------------------------
SELECT
    a.category_id,
    c.category_name,
    COUNT(*) AS cnt
FROM articles a
LEFT JOIN categories c ON a.category_id = c.category_id
WHERE a.is_fake = 0
GROUP BY a.category_id, c.category_name
ORDER BY cnt DESC;


-- ------------------------------------------------
-- 4. 중복 검증
-- ------------------------------------------------

-- 4-1. 동일 원본에서 복수 생성된 가짜 기사
SELECT url, COUNT(*) AS fake_cnt
FROM articles
WHERE is_fake = 1 AND url IS NOT NULL
GROUP BY url
HAVING COUNT(*) > 1
ORDER BY fake_cnt DESC
LIMIT 20;

-- 4-2. 본문 완전 중복
SELECT
    COUNT(*)                AS total,
    COUNT(DISTINCT content) AS distinct_content,
    COUNT(*) - COUNT(DISTINCT content) AS duplicated
FROM articles
WHERE is_fake = 1;

-- 4-3. 진짜 기사 URL 중복
SELECT url, COUNT(*) AS cnt
FROM articles
WHERE is_fake = 0 AND url IS NOT NULL
GROUP BY url
HAVING COUNT(*) > 1
LIMIT 20;


-- ------------------------------------------------
-- 5. 수집 품질 검증
-- ------------------------------------------------

-- 5-1. 발행일이 수집 시각으로 대체된 건수 (pubDate 폴백)
SELECT COUNT(*) AS fallback_cnt
FROM articles
WHERE is_fake = 0
  AND ABS(TIMESTAMPDIFF(MINUTE, published_at, created_at)) < 5;

-- 5-2. JTBC 수집 건수
--      API 수집 워크플로의 노드 참조 오류로 누락 가능성 있음
SELECT COUNT(*) AS jtbc_cnt
FROM articles
WHERE is_fake = 0 AND url LIKE '%jtbc%';

-- 5-3. 본문 길이 절단 의심 건수 (21,000자 상한)
SELECT COUNT(*) AS truncated_cnt
FROM articles
WHERE CHAR_LENGTH(content) >= 20900;

-- 5-4. 발행일 범위
SELECT
    MIN(published_at) AS earliest,
    MAX(published_at) AS latest,
    COUNT(*)          AS total
FROM articles
WHERE is_fake = 0;

-- 5-5. 월별 수집 분포
SELECT
    DATE_FORMAT(published_at, '%Y-%m') AS month,
    COUNT(*) AS cnt
FROM articles
WHERE is_fake = 0
GROUP BY month
ORDER BY month;


-- ------------------------------------------------
-- 6. 본문 길이 분포
-- ------------------------------------------------
SELECT
    is_fake,
    COUNT(*)                      AS cnt,
    MIN(CHAR_LENGTH(content))     AS min_len,
    ROUND(AVG(CHAR_LENGTH(content))) AS avg_len,
    MAX(CHAR_LENGTH(content))     AS max_len
FROM articles
GROUP BY is_fake;

-- 유형별 길이 (원문 대비 생성문 길이 비교용)
SELECT
    prompt_type,
    COUNT(*)                         AS cnt,
    ROUND(AVG(CHAR_LENGTH(content))) AS avg_len
FROM articles
WHERE is_fake = 1
GROUP BY prompt_type
ORDER BY prompt_type;


-- ------------------------------------------------
-- 7. 생성 품질 검증
-- ------------------------------------------------

-- 7-1. 유형 5(동사 반전) 조작 강도 확인
--      원문과 길이가 거의 같은 것은 정상이나,
--      내용까지 동일하면 조작이 적용되지 않은 것이다.
SELECT COUNT(*) AS suspicious_cnt
FROM articles f
JOIN articles o ON f.url = o.url AND o.is_fake = 0
WHERE f.is_fake = 1
  AND f.prompt_type = 5
  AND f.content = o.content;

-- 7-2. LLM 거부 응답이 저장된 건수
SELECT COUNT(*) AS refused_cnt
FROM articles
WHERE is_fake = 1
  AND (content LIKE '%죄송합니다%'
    OR content LIKE '%도와드릴 수 없%'
    OR content LIKE '%I cannot%'
    OR content LIKE '%I''m sorry%');
