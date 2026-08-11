-- =====================================================================
-- 02_constraints_indexes.sql — keys, data-quality checks and indexes.
-- Run AFTER the bulk load (building a B-tree once on sorted data beats
-- maintaining it row by row during COPY).
-- =====================================================================

-- Primary keys
ALTER TABLE raw.aisles       ADD CONSTRAINT pk_aisles       PRIMARY KEY (aisle_id);
ALTER TABLE raw.departments  ADD CONSTRAINT pk_departments  PRIMARY KEY (department_id);
ALTER TABLE raw.products     ADD CONSTRAINT pk_products     PRIMARY KEY (product_id);
ALTER TABLE raw.orders       ADD CONSTRAINT pk_orders       PRIMARY KEY (order_id);
ALTER TABLE raw.order_products_prior ADD CONSTRAINT pk_op_prior PRIMARY KEY (order_id, product_id);
ALTER TABLE raw.order_products_train ADD CONSTRAINT pk_op_train PRIMARY KEY (order_id, product_id);

-- Domain checks
ALTER TABLE raw.orders ADD CONSTRAINT ck_eval_set CHECK (eval_set IN ('prior', 'train', 'test'));
ALTER TABLE raw.orders ADD CONSTRAINT ck_dow      CHECK (order_dow BETWEEN 0 AND 6);
ALTER TABLE raw.orders ADD CONSTRAINT ck_hour     CHECK (order_hour_of_day BETWEEN 0 AND 23);
ALTER TABLE raw.order_products_prior ADD CONSTRAINT ck_reordered_p CHECK (reordered IN (0, 1));
ALTER TABLE raw.order_products_train ADD CONSTRAINT ck_reordered_t CHECK (reordered IN (0, 1));

-- Foreign keys (validated once, in bulk)
ALTER TABLE raw.products ADD CONSTRAINT fk_products_aisle FOREIGN KEY (aisle_id) REFERENCES raw.aisles;
ALTER TABLE raw.products ADD CONSTRAINT fk_products_dept  FOREIGN KEY (department_id) REFERENCES raw.departments;
ALTER TABLE raw.order_products_prior ADD CONSTRAINT fk_opp_order   FOREIGN KEY (order_id)   REFERENCES raw.orders;
ALTER TABLE raw.order_products_prior ADD CONSTRAINT fk_opp_product FOREIGN KEY (product_id) REFERENCES raw.products;
ALTER TABLE raw.order_products_train ADD CONSTRAINT fk_opt_order   FOREIGN KEY (order_id)   REFERENCES raw.orders;
ALTER TABLE raw.order_products_train ADD CONSTRAINT fk_opt_product FOREIGN KEY (product_id) REFERENCES raw.products;

-- Secondary indexes for access paths the API uses
-- (user -> orders, newest first) is a composite index so the
-- "last N orders of a user" query is an index-only range scan.
CREATE INDEX ix_orders_user_ordernum ON raw.orders (user_id, order_number DESC);

ANALYZE raw.aisles, raw.departments, raw.products, raw.orders,
        raw.order_products_prior, raw.order_products_train;
