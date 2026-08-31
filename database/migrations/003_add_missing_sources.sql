-- ================================================
-- 마이그레이션 003 — 누락 언론사 추가
--
-- 배경: 초기 시드에 동아일보와 JTBC 가 없었으나, 두 언론사는
--       수집 대상에 포함되어 있었다. articles.source_id 에
--       외래키 제약이 걸려 있어 해당 언론사 기사 INSERT 가
--       실패하는 문제가 있었다.
--
-- 주의: source_id 값은 n8n 워크플로에 하드코딩된 값과
--       일치해야 하므로 명시적으로 지정한다.
-- ================================================

USE fake_news_db;

INSERT INTO news_sources (source_id, source_name, base_url) VALUES
    (7,  '동아일보', 'https://www.donga.com'),
    (10, 'JTBC',    'https://news.jtbc.co.kr')
ON DUPLICATE KEY UPDATE
    source_name = VALUES(source_name),
    base_url    = VALUES(base_url);
