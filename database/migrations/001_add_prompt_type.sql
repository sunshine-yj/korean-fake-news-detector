-- ================================================
-- 마이그레이션 001 — prompt_type 컬럼 추가
--
-- 배경: 초기 DDL 에는 조작 유형을 기록하는 컬럼이 없었다.
--       가짜뉴스 생성 파이프라인 도입 시점에 추가되었다.
-- ================================================

USE fake_news_db;

ALTER TABLE articles
    ADD COLUMN prompt_type TINYINT NULL
        COMMENT '조작 유형 1~5 (진짜 기사는 NULL)'
        AFTER is_fake;

ALTER TABLE articles
    ADD INDEX idx_prompt_type (prompt_type),
    ADD INDEX idx_fake_prompt (is_fake, prompt_type);
