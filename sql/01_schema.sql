-- =====================================================================
-- 01_schema.sql — raw tables, created WITHOUT indexes/constraints.
-- Bulk COPY into bare heap tables is much faster; keys and indexes are
-- added afterwards in 02_constraints_indexes.sql.
-- =====================================================================
DROP SCHEMA IF EXISTS feat CASCADE;
DROP SCHEMA IF EXISTS raw CASCADE;
CREATE SCHEMA raw;

CREATE TABLE raw.aisles (
    aisle_id    integer NOT NULL,
    aisle       text    NOT NULL
);

CREATE TABLE raw.departments (
    department_id integer NOT NULL,
    department    text    NOT NULL
);

CREATE TABLE raw.products (
    product_id    integer NOT NULL,
    product_name  text    NOT NULL,
    aisle_id      integer NOT NULL,
    department_id integer NOT NULL
);

CREATE TABLE raw.orders (
    order_id               integer  NOT NULL,
    user_id                integer  NOT NULL,
    eval_set               text     NOT NULL,
    order_number           smallint NOT NULL,
    order_dow              smallint NOT NULL,
    order_hour_of_day      smallint NOT NULL,
    days_since_prior_order real              -- NULL for a user's first order
);

CREATE TABLE raw.order_products_prior (
    order_id          integer  NOT NULL,
    product_id        integer  NOT NULL,
    add_to_cart_order smallint NOT NULL,
    reordered         smallint NOT NULL
);

CREATE TABLE raw.order_products_train (LIKE raw.order_products_prior);
