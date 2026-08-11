-- =====================================================================
-- 03_features.sql — the feature store, built entirely in Postgres.
--
-- Every feature is computed from PRIOR orders only, so nothing about the
-- order we are predicting leaks into the inputs.
--
-- Layers:
--   feat.prior_orders     order-level: recency rank + cumulative day clock
--   feat.target_orders    the one order per user we predict (train/test)
--   feat.up_events        one row per (user, product, prior order) purchase
--   feat.user_features    user behaviour
--   feat.user_recent_gaps last-20 inter-order gaps (input to the GRU)
--   feat.product_features product popularity / loyalty
--   feat.up_features      user x product history (+ 20-order purchase bitmask)
--   feat.user_aisle / feat.user_department   category affinity shares
--   feat.labels           ground truth from the train orders
--   feat.user_split       deterministic train/valid/test split by user
--   feat.candidates       ONE view used by both training and the API
-- =====================================================================
DROP SCHEMA IF EXISTS feat CASCADE;
CREATE SCHEMA feat;

-- ---------------------------------------------------------------------
-- Order-level context.
--   day_clock    : running total of days since the user's first order,
--                  so "days between two purchases" becomes a subtraction.
--   recency_rank : 1 = most recent prior order.
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.prior_orders AS
SELECT
    o.order_id,
    o.user_id,
    o.order_number,
    o.order_dow,
    o.order_hour_of_day,
    COALESCE(o.days_since_prior_order, 0)::float8                          AS gap_days,
    SUM(COALESCE(o.days_since_prior_order, 0)::float8)
        OVER (PARTITION BY o.user_id ORDER BY o.order_number)              AS day_clock,
    ROW_NUMBER() OVER (PARTITION BY o.user_id ORDER BY o.order_number DESC) AS recency_rank,
    COUNT(*)     OVER (PARTITION BY o.user_id)                             AS n_orders
FROM raw.orders o
WHERE o.eval_set = 'prior';

CREATE UNIQUE INDEX ON feat.prior_orders (order_id);
CREATE INDEX ON feat.prior_orders (user_id, recency_rank);

-- ---------------------------------------------------------------------
-- The order we want to predict (exactly one per user).
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.target_orders AS
SELECT
    user_id,
    order_id,
    eval_set,
    order_number,
    order_dow,
    order_hour_of_day,
    days_since_prior_order::float8 AS days_since_last
FROM raw.orders
WHERE eval_set IN ('train', 'test');

CREATE UNIQUE INDEX ON feat.target_orders (user_id);

-- ---------------------------------------------------------------------
-- Purchase events: the 32M-row prior basket table joined to order context.
-- Materialised once because every downstream aggregate reads it.
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.up_events AS
SELECT
    po.user_id,
    op.product_id,
    po.order_number,
    po.recency_rank,
    po.day_clock,
    po.n_orders,
    op.add_to_cart_order,
    op.reordered
FROM raw.order_products_prior op
JOIN feat.prior_orders po USING (order_id);

-- ---------------------------------------------------------------------
-- User x product features.
--   LAG gives the previous purchase of the same item -> re-buy interval.
--   up_bitmask: bit (k-1) is set if the item was in the user's k-th most
--   recent order (k <= 20). The NN decodes it into a 20-step sequence.
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.up_features AS
WITH ev AS (
    SELECT
        e.*,
        LAG(e.day_clock) OVER (PARTITION BY e.user_id, e.product_id
                               ORDER BY e.order_number) AS prev_clock
    FROM feat.up_events e
)
SELECT
    user_id,
    product_id,
    COUNT(*)                                                  AS up_orders,
    MIN(order_number)                                         AS up_first_order,
    MIN(recency_rank) - 1                                     AS up_orders_since_last,
    MAX(day_clock)                                            AS up_last_clock,
    AVG(add_to_cart_order::float8)                            AS up_avg_cart_pos,
    AVG(day_clock - prev_clock)                               AS up_avg_interval,
    SUM(CASE WHEN recency_rank <= 5 THEN 1 ELSE 0 END)        AS up_last5,
    SUM(CASE WHEN recency_rank <= 20
             THEN (1::bigint << (recency_rank - 1)::int)
             ELSE 0 END)::bigint                              AS up_bitmask
FROM ev
GROUP BY user_id, product_id;

CREATE UNIQUE INDEX ON feat.up_features (user_id, product_id);

-- ---------------------------------------------------------------------
-- User features (built from up_features, so it comes after it).
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.user_features AS
WITH ord AS (
    SELECT
        user_id,
        MAX(n_orders)                                         AS n_orders,
        AVG(gap_days)         FILTER (WHERE order_number > 1) AS avg_gap,
        STDDEV_SAMP(gap_days) FILTER (WHERE order_number > 1) AS std_gap,
        MAX(day_clock)                                        AS total_days,
        AVG(order_hour_of_day::float8)                        AS avg_hour
    FROM feat.prior_orders
    GROUP BY user_id
),
items AS (
    -- Reuse the (user, product) aggregate instead of re-scanning 32M rows:
    --   distinct products = rows, items = SUM(up_orders),
    --   reorders = every purchase after the first = SUM(up_orders - 1).
    -- reorder_ratio = reorders / items bought in orders 2..n
    --   (items in order 1 = products whose first purchase was order 1).
    SELECT
        user_id,
        SUM(up_orders)                                                AS n_items,
        COUNT(*)                                                      AS n_distinct,
        SUM(up_orders - 1)::float8
            / NULLIF(SUM(up_orders) - SUM((up_first_order = 1)::int), 0) AS reorder_ratio
    FROM feat.up_features
    GROUP BY user_id
)
SELECT
    o.user_id,
    o.n_orders,
    o.avg_gap,
    COALESCE(o.std_gap, 0)               AS std_gap,
    o.total_days,
    o.avg_hour,
    i.n_items,
    i.n_items::float8 / o.n_orders       AS avg_basket,
    i.n_distinct,
    i.n_distinct::float8 / i.n_items     AS distinct_ratio,
    COALESCE(i.reorder_ratio, 0)         AS reorder_ratio
FROM ord o
JOIN items i USING (user_id);

CREATE UNIQUE INDEX ON feat.user_features (user_id);

-- ---------------------------------------------------------------------
-- Last 20 inter-order gaps per user, oldest -> newest (GRU timeline).
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.user_recent_gaps AS
SELECT
    user_id,
    ARRAY_AGG(gap_days ORDER BY order_number) AS gaps_last20
FROM feat.prior_orders
WHERE recency_rank <= 20
GROUP BY user_id;

CREATE UNIQUE INDEX ON feat.user_recent_gaps (user_id);

-- ---------------------------------------------------------------------
-- Product features.
--   repeat_user_share : of the people who tried it, how many came back?
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.product_features AS
WITH users AS (
    SELECT
        product_id,
        COUNT(*)                              AS n_users,
        AVG((up_orders >= 2)::int::float8)    AS repeat_user_share,
        AVG(up_orders::float8)                AS avg_buys_per_user
    FROM feat.up_features
    GROUP BY product_id
),
sales AS (
    SELECT
        product_id,
        COUNT(*)                            AS n_purchases,
        AVG(reordered::float8)              AS reorder_rate,
        AVG(add_to_cart_order::float8)      AS avg_cart_pos
    FROM feat.up_events
    GROUP BY product_id
)
SELECT
    p.product_id,
    p.aisle_id,
    p.department_id,
    COALESCE(s.n_purchases, 0)          AS p_purchases,
    COALESCE(s.reorder_rate, 0)         AS p_reorder_rate,
    COALESCE(s.avg_cart_pos, 0)         AS p_avg_cart_pos,
    COALESCE(u.n_users, 0)              AS p_users,
    COALESCE(u.repeat_user_share, 0)    AS p_repeat_user_share,
    COALESCE(u.avg_buys_per_user, 0)    AS p_avg_buys_per_user
FROM raw.products p
LEFT JOIN sales s USING (product_id)
LEFT JOIN users u USING (product_id);

CREATE UNIQUE INDEX ON feat.product_features (product_id);

-- ---------------------------------------------------------------------
-- Category affinity: share of a user's items from each aisle / department.
-- A window over an aggregate gives the per-user denominator in one pass.
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.user_aisle AS
SELECT
    e.user_id,
    p.aisle_id,
    COUNT(*)::float8 / SUM(COUNT(*)) OVER (PARTITION BY e.user_id) AS ua_share
FROM feat.up_events e
JOIN raw.products p USING (product_id)
GROUP BY e.user_id, p.aisle_id;

CREATE UNIQUE INDEX ON feat.user_aisle (user_id, aisle_id);

CREATE MATERIALIZED VIEW feat.user_department AS
SELECT
    e.user_id,
    p.department_id,
    COUNT(*)::float8 / SUM(COUNT(*)) OVER (PARTITION BY e.user_id) AS ud_share
FROM feat.up_events e
JOIN raw.products p USING (product_id)
GROUP BY e.user_id, p.department_id;

CREATE UNIQUE INDEX ON feat.user_department (user_id, department_id);

-- ---------------------------------------------------------------------
-- Labels: items from a user's prior history that appear in the train order.
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.labels AS
SELECT o.user_id, opt.product_id, 1 AS label
FROM raw.order_products_train opt
JOIN raw.orders o USING (order_id)
WHERE opt.reordered = 1;

CREATE UNIQUE INDEX ON feat.labels (user_id, product_id);

-- ---------------------------------------------------------------------
-- Deterministic user-level split (no user appears in two splits).
-- Kaggle "test" users have no labels: they're only served by the API.
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW feat.user_split AS
SELECT
    user_id,
    CASE
        WHEN eval_set = 'test'                          THEN 'unlabeled'
        WHEN ABS(HASHTEXT(user_id::text)) % 10 < 7      THEN 'train'
        WHEN ABS(HASHTEXT(user_id::text)) % 10 < 8      THEN 'valid'
        ELSE 'test'
    END AS split
FROM feat.target_orders;

CREATE UNIQUE INDEX ON feat.user_split (user_id);

-- ---------------------------------------------------------------------
-- feat.candidates — the single source of truth for model inputs.
-- Training reads it in bulk; the API reads it WHERE user_id = $1.
-- Same SQL on both paths = no training/serving skew.
-- ---------------------------------------------------------------------
CREATE VIEW feat.candidates AS
SELECT
    up.user_id,
    up.product_id,
    p.aisle_id,
    p.department_id,

    -- user
    u.n_orders                       AS u_n_orders,
    u.avg_gap                        AS u_avg_gap,
    u.std_gap                        AS u_std_gap,
    u.avg_basket                     AS u_avg_basket,
    u.n_distinct                     AS u_n_distinct,
    u.distinct_ratio                 AS u_distinct_ratio,
    u.reorder_ratio                  AS u_reorder_ratio,
    u.avg_hour                       AS u_avg_hour,

    -- the order being predicted (context known at order time)
    t.days_since_last                AS t_days_since_last,
    t.order_dow                      AS t_dow,
    t.order_hour_of_day              AS t_hour,

    -- product
    p.p_purchases,
    p.p_reorder_rate,
    p.p_avg_cart_pos,
    p.p_users,
    p.p_repeat_user_share,
    p.p_avg_buys_per_user,

    -- user x product
    up.up_orders,
    up.up_orders::float8 / u.n_orders                               AS up_order_rate,
    up.up_orders::float8 / (u.n_orders - up.up_first_order + 1)     AS up_rate_since_first,
    up.up_orders_since_last,
    u.total_days - up.up_last_clock                                 AS up_days_since_last,
    COALESCE(up.up_avg_interval, -1)                                AS up_avg_interval,
    up.up_avg_cart_pos,
    up.up_last5,
    up.up_last5::float8 / LEAST(u.n_orders, 5)                      AS up_last5_rate,
    COALESCE(ua.ua_share, 0)                                        AS ua_share,
    COALESCE(ud.ud_share, 0)                                        AS ud_share,

    -- raw sequence inputs for the neural net
    up.up_bitmask,
    g.gaps_last20
FROM feat.up_features      up
JOIN feat.user_features    u  USING (user_id)
JOIN feat.target_orders    t  USING (user_id)
JOIN feat.user_recent_gaps g  USING (user_id)
JOIN feat.product_features p  ON p.product_id = up.product_id
LEFT JOIN feat.user_aisle      ua ON ua.user_id = up.user_id AND ua.aisle_id = p.aisle_id
LEFT JOIN feat.user_department ud ON ud.user_id = up.user_id AND ud.department_id = p.department_id;

ANALYZE;
