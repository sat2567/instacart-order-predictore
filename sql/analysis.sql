-- Business-facing EDA queries. Results are written to artifacts/analysis.md
-- by `python -m reorder.pipeline analysis`.

-- name: Reorder rate by department
SELECT d.department,
       COUNT(*)                                   AS items,
       ROUND(AVG(op.reordered)::numeric, 3)       AS reorder_rate
FROM raw.order_products_prior op
JOIN raw.products p    USING (product_id)
JOIN raw.departments d USING (department_id)
GROUP BY d.department
ORDER BY reorder_rate DESC
LIMIT 10;

-- name: Most "sticky" products (min 2,000 buyers), share of buyers who came back
SELECT pr.product_name, pf.p_users AS buyers,
       ROUND(pf.p_repeat_user_share::numeric, 3) AS repeat_share
FROM feat.product_features pf
JOIN raw.products pr USING (product_id)
WHERE pf.p_users >= 2000
ORDER BY pf.p_repeat_user_share DESC
LIMIT 10;

-- name: Basket size by hour of day
SELECT o.order_hour_of_day AS hour,
       COUNT(DISTINCT o.order_id)                                   AS orders,
       ROUND(COUNT(*)::numeric / COUNT(DISTINCT o.order_id), 2)     AS avg_basket
FROM raw.orders o
JOIN raw.order_products_prior op USING (order_id)
GROUP BY o.order_hour_of_day
ORDER BY hour;

-- name: Reorder rate vs. days since the previous order
SELECT CASE
         WHEN o.days_since_prior_order <= 7  THEN '0-7'
         WHEN o.days_since_prior_order <= 14 THEN '8-14'
         WHEN o.days_since_prior_order <= 29 THEN '15-29'
         ELSE '30 (capped)'
       END                                        AS days_since_prior,
       COUNT(*)                                   AS items,
       ROUND(AVG(op.reordered)::numeric, 3)       AS reorder_rate
FROM raw.orders o
JOIN raw.order_products_prior op USING (order_id)
WHERE o.days_since_prior_order IS NOT NULL
GROUP BY 1
ORDER BY MIN(o.days_since_prior_order);

-- name: Label base rate in the target orders
SELECT s.split,
       COUNT(*)                                       AS candidates,
       ROUND(AVG(COALESCE(l.label, 0))::numeric, 4)   AS positive_rate
FROM feat.candidates c
JOIN feat.user_split s USING (user_id)
LEFT JOIN feat.labels l USING (user_id, product_id)
WHERE s.split <> 'unlabeled'
GROUP BY s.split
ORDER BY s.split;
