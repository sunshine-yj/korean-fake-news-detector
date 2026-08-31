-- ================================================
-- 마이그레이션 002 — 전처리 텍스트 컬럼 추가
--
-- 배경: 원문(content)과 모델 입력용 정제 텍스트를 분리해
--       전처리 로직 변경 시 원문을 보존할 수 있도록 했다.
-- ================================================

USE fake_news_db;

ALTER TABLE articles
    ADD COLUMN title_clean VARCHAR(500) NULL
        COMMENT '전처리된 제목 (모델 입력)'
        AFTER content,
    ADD COLUMN content_clean TEXT NULL
        COMMENT '전처리된 본문 (모델 입력)'
        AFTER title_clean;
