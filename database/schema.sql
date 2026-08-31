-- ================================================
-- 가짜뉴스 판별 시스템 MySQL DDL
--
-- 주의: 본 스크립트는 프로젝트 종료 시점 기준으로 정리한 것이다.
--       실제 데이터 구축은 초기 DDL로 시작해 운영 중 ALTER TABLE 로
--       컬럼을 추가하며 진행되었으므로, 수집 시점별 스키마가 다르다.
--       변경 이력은 migrations/ 및 README.md 참조.
--
-- 실행: mysql -u root -p < schema.sql
-- ================================================

DROP DATABASE IF EXISTS fake_news_db;

CREATE DATABASE fake_news_db
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE fake_news_db;


-- ------------------------------------------------
-- 1. NEWS_SOURCES (뉴스 출처)
-- ------------------------------------------------
CREATE TABLE news_sources (
    source_id   INT           NOT NULL AUTO_INCREMENT,
    source_name VARCHAR(100)  NOT NULL COMMENT '출처 이름',
    base_url    VARCHAR(300)           COMMENT '출처 사이트 주소',
    domain      VARCHAR(100)           COMMENT 'URL 매칭용 도메인 (예: yna.co.kr)',
    created_at  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (source_id),
    INDEX idx_domain (domain)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------
-- 2. CATEGORIES (카테고리)
-- ------------------------------------------------
CREATE TABLE categories (
    category_id   INT          NOT NULL AUTO_INCREMENT,
    category_name VARCHAR(50)  NOT NULL COMMENT '카테고리 이름',
    description   VARCHAR(200)          COMMENT '카테고리 설명',

    PRIMARY KEY (category_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------
-- 3. ARTICLES (기사)
-- ------------------------------------------------
CREATE TABLE articles (
    article_id        BIGINT        NOT NULL AUTO_INCREMENT,
    title             VARCHAR(500)  NOT NULL COMMENT '기사 제목',
    content           TEXT          NOT NULL COMMENT '기사 본문 (최대 21,000자로 절단 저장)',
    title_clean       VARCHAR(500)           COMMENT '전처리된 제목 (모델 입력)',
    content_clean     TEXT                   COMMENT '전처리된 본문 (모델 입력)',
    url               VARCHAR(1000)          COMMENT '기사 원문 URL (가짜 기사는 원본 URL 복사)',
    published_at      DATETIME               COMMENT '기사 발행 일시 (pubDate 부재 시 수집 시각)',

    is_fake           TINYINT       NOT NULL COMMENT '라벨 (0: 진짜, 1: 가짜)',
    prompt_type       TINYINT                COMMENT '조작 유형 1~5 (진짜 기사는 NULL)',
    origin_article_id BIGINT                 COMMENT '가짜 기사의 원본 article_id',

    source_id         INT                    COMMENT '출처 ID',
    category_id       INT                    COMMENT '카테고리 ID',
    created_at        TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (article_id),

    FOREIGN KEY (source_id)         REFERENCES news_sources(source_id) ON DELETE SET NULL,
    FOREIGN KEY (category_id)       REFERENCES categories(category_id) ON DELETE SET NULL,
    FOREIGN KEY (origin_article_id) REFERENCES articles(article_id)    ON DELETE SET NULL,

    INDEX idx_is_fake         (is_fake),
    INDEX idx_prompt_type     (prompt_type),
    INDEX idx_fake_prompt     (is_fake, prompt_type),
    INDEX idx_source_id       (source_id),
    INDEX idx_category_id     (category_id),
    INDEX idx_published_at    (published_at),
    INDEX idx_origin          (origin_article_id),
    INDEX idx_url             (url(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 참고: 실제 구축 시에는 url 에 UNIQUE 제약이 없었고, 중복 방지를
--       n8n 워크플로의 SELECT COUNT(*) + IF 노드로 처리했다.
--       재구축 시에는 아래 제약을 추가하는 편이 안전하다.
--
--   ALTER TABLE articles ADD UNIQUE KEY uk_url (url(255));
--
--   단, 가짜 기사가 원본의 url 을 복사해 저장하는 구조이므로
--   이 제약을 걸려면 가짜 기사의 url 을 NULL 로 두거나
--   origin_url 컬럼으로 분리해야 한다.


-- ------------------------------------------------
-- 4. 기초 데이터
--    source_id 는 n8n 파이프라인의 하드코딩 값과 일치해야 한다.
--    (자동 증가에 의존하지 않고 명시적으로 지정)
-- ------------------------------------------------
INSERT INTO news_sources (source_id, source_name, base_url, domain) VALUES
    (1,  '연합뉴스',   'https://www.yna.co.kr',        'yna.co.kr'),
    (2,  '노컷뉴스',   'https://www.nocutnews.co.kr',  'nocutnews.co.kr'),
    (3,  'MBC',       'https://imnews.imbc.com',      'imbc.com'),
    (4,  'SBS',       'https://news.sbs.co.kr',       'sbs.co.kr'),
    (5,  '조선일보',   'https://www.chosun.com',       'chosun.com'),
    (6,  '한겨레',     'https://www.hani.co.kr',       'hani.co.kr'),
    (7,  '동아일보',   'https://www.donga.com',        'donga.com'),
    (8,  '네이버뉴스', 'https://news.naver.com',       'news.naver.com'),
    (9,  'GPT-4o',    NULL,                           NULL),
    (10, 'JTBC',      'https://news.jtbc.co.kr',      'jtbc.co.kr'),
    (11, 'GPT-5-mini', NULL,                          NULL);

INSERT INTO categories (category_id, category_name, description) VALUES
    (1, '정치', '정치 관련 뉴스'),
    (2, '경제', '경제 및 금융 관련 뉴스'),
    (3, '사회', '사회 및 생활 관련 뉴스'),
    (4, '문화', '문화 및 연예 관련 뉴스'),
    (5, '국제', '해외 및 국제 관련 뉴스');


-- ------------------------------------------------
-- 5. 계정 생성 (필요 시 주석 해제)
--    비밀번호는 반드시 직접 지정할 것
-- ------------------------------------------------
-- CREATE USER 'fakenews_user'@'localhost' IDENTIFIED BY 'CHANGE_ME';
-- GRANT ALL PRIVILEGES ON fake_news_db.* TO 'fakenews_user'@'localhost';
-- FLUSH PRIVILEGES;


-- ------------------------------------------------
-- 6. 생성 확인
-- ------------------------------------------------
SHOW TABLES;
SELECT * FROM news_sources;
SELECT * FROM categories;
