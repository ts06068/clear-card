CREATE TABLE paper (
    paper_id BIGINT PRIMARY KEY,
    eid NVARCHAR(128) NULL,
    [dc:title] NVARCHAR(MAX) NULL,
    [prism:doi] NVARCHAR(255) NULL,
    [pubmed-id] NVARCHAR(64) NULL,
    [prism:coverDate] DATE NULL,
    [prism:publicationName] NVARCHAR(512) NULL
);

CREATE TABLE citation (
    paper_id BIGINT PRIMARY KEY,
    eid NVARCHAR(128) NULL,
    [dc:title] NVARCHAR(MAX) NULL,
    [prism:doi] NVARCHAR(255) NULL,
    [pubmed-id] NVARCHAR(64) NULL,
    [prism:coverDate] DATE NULL,
    [prism:publicationName] NVARCHAR(512) NULL
);

CREATE TABLE relationship (
    paper_id_1 BIGINT NOT NULL,
    paper_id_2 BIGINT NOT NULL,
    relationship NVARCHAR(32) NOT NULL,
    FOREIGN KEY (paper_id_1) REFERENCES paper(paper_id),
    FOREIGN KEY (paper_id_2) REFERENCES citation(paper_id)
);
