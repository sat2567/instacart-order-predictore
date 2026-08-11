-- name: user_order_history
-- The API's "last 5 orders of a user, with product names" lookup.
SELECT o.order_number, o.order_dow, o.order_hour_of_day, p.product_name
FROM raw.orders o
JOIN raw.order_products_prior op ON op.order_id = o.order_id
JOIN raw.products p             ON p.product_id = op.product_id
WHERE o.user_id = 4242
  AND o.order_number > (SELECT MAX(order_number) - 5 FROM raw.orders WHERE user_id = 4242 AND eval_set = 'prior')
  AND o.eval_set = 'prior'
ORDER BY o.order_number DESC, op.add_to_cart_order;

-- name: single_order_basket
-- Point lookup of one basket.
SELECT op.product_id, op.add_to_cart_order, op.reordered
FROM raw.order_products_prior op
WHERE op.order_id = 2539329;

-- name: user_reorder_rate
-- Per-user reorder rate straight from the raw tables.
SELECT AVG(op.reordered::float8)
FROM raw.orders o
JOIN raw.order_products_prior op ON op.order_id = o.order_id
WHERE o.user_id = 4242 AND o.order_number > 1;
