-- Keep target-to-source citation direction consistent with the analysis pipeline.
SELECT
    r.paper_id_1,
    r.paper_id_2,
    r.relationship,
    c.[dc:title] AS source_title,
    c.eid AS source_eid,
    c.[prism:doi] AS source_doi,
    c.[pubmed-id] AS source_pubmed_id,
    YEAR(c.[prism:coverDate]) AS source_pub_year,
    c.[prism:coverDate] AS source_pub_date,
    c.[prism:publicationName] AS source_journal,
    p.[dc:title] AS target_title,
    p.eid AS target_eid,
    p.[prism:doi] AS target_doi,
    p.[pubmed-id] AS target_pubmed_id,
    YEAR(p.[prism:coverDate]) AS target_pub_year,
    p.[prism:coverDate] AS target_pub_date,
    p.[prism:publicationName] AS target_journal
FROM paper AS p
LEFT JOIN relationship AS r
    ON p.paper_id = r.paper_id_1
    AND r.relationship = 'citing'
LEFT JOIN citation AS c
    ON r.paper_id_2 = c.paper_id
ORDER BY p.[prism:coverDate], p.[prism:publicationName], p.[prism:doi];
