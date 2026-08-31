-- ================================================
-- 마이그레이션 004 — 원본 기사 참조 컬럼 추가
--
-- 배경: 가짜 기사가 어떤 원본에서 생성되었는지 기록하는
--       컬럼이 없어, 원본-가짜 쌍의 추적이 불가능했다.
--
-- 주의: 본 프로젝트의 데이터 구축 완료 이후 정의한 컬럼이다.
--       기존 레코드는 값이 NULL 이며, url 을 통한 간접 연결만
--       가능하다. 아래 백필 쿼리로 부분 복원할 수 있다.
-- ================================================

USE fake_news_db;

ALTER TABLE articles
    ADD COLUMN origin_article_id BIGINT NULL
        COMMENT '가짜 기사의 원본 article_id'
        AFTER prompt_type;

ALTER TABLE articles
    ADD INDEX idx_origin (origin_article_id);

ALTER TABLE articles
    ADD CONSTRAINT fk_origin_article
        FOREIGN KEY (origin_article_id) REFERENCES articles(article_id)
        ON DELETE SET NULL;


-- ------------------------------------------------
-- 백필 (선택) — url 매칭으로 기존 데이터 복원
--
-- 가짜 기사가 원본의 url 을 복사해 저장하는 구조를 이용한다.
-- 동일 url 의 진짜 기사가 여러 건이면 최소 article_id 를 택한다.
-- ------------------------------------------------
-- UPDATE articles f
-- JOIN (
--     SELECT url, MIN(article_id) AS origin_id
--     FROM articles
--     WHERE is_fake = 0 AND url IS NOT NULL
--     GROUP BY url
-- ) o ON f.url = o.url
-- SET f.origin_article_id = o.origin_id
-- WHERE f.is_fake = 1 AND f.origin_article_id IS NULL;
