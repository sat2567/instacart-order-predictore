# SQL analysis (auto-generated)

## Reorder rate by department

| department   |   items |   reorder_rate |
|:-------------|--------:|---------------:|
| dairy eggs   | 5414016 |          0.67  |
| beverages    | 2690129 |          0.653 |
| produce      | 9479291 |          0.65  |
| bakery       | 1176787 |          0.628 |
| deli         | 1051249 |          0.608 |
| pets         |   97724 |          0.601 |
| babies       |  423802 |          0.579 |
| bulk         |   34573 |          0.577 |
| snacks       | 2887550 |          0.574 |
| alcohol      |  153696 |          0.57  |

## Most "sticky" products (min 2,000 buyers), share of buyers who came back

| product_name                        |   buyers |   repeat_share |
|:------------------------------------|---------:|---------------:|
| Banana                              |    73956 |          0.746 |
| Milk, Organic, Vitamin D            |     2942 |          0.72  |
| Bag of Organic Bananas              |    63537 |          0.712 |
| Fresh CA Grown Eggs                 |     2149 |          0.698 |
| Organic Reduced Fat Milk            |     5325 |          0.698 |
| Organic Whole Milk with DHA Omega-3 |     2428 |          0.685 |
| Organic Lowfat 1% Milk              |     2381 |          0.681 |
| Organic Hass Avocado                |    43453 |          0.668 |
| Organic Whole Milk                  |    23395 |          0.667 |
| Organic Baby Spinach                |    55037 |          0.664 |

## Basket size by hour of day

|   hour |   orders |   avg_basket |
|-------:|---------:|-------------:|
|      0 |    21372 |        10.24 |
|      1 |    11596 |         9.98 |
|      2 |     7070 |         9.82 |
|      3 |     5120 |        10.02 |
|      4 |     5175 |        10.3  |
|      5 |     8972 |         9.82 |
|      6 |    28792 |        10.1  |
|      7 |    86656 |        10.29 |
|      8 |   168321 |        10.22 |
|      9 |   243496 |        10.09 |
|     10 |   271885 |        10.17 |
|     11 |   268006 |        10.22 |
|     12 |   256206 |        10.23 |
|     13 |   261174 |        10.2  |
|     14 |   265556 |        10.14 |
|     15 |   266132 |        10.01 |
|     16 |   255949 |         9.91 |
|     17 |   214080 |         9.76 |
|     18 |   170998 |         9.58 |
|     19 |   131620 |         9.57 |
|     20 |    98109 |         9.96 |
|     21 |    73436 |        10.84 |
|     22 |    57540 |        11.03 |
|     23 |    37613 |        10.7  |

## Reorder rate vs. days since the previous order

| days_since_prior   |    items |   reorder_rate |
|:-------------------|---------:|---------------:|
| 0-7                | 14938993 |          0.675 |
| 8-14               |  7599808 |          0.644 |
| 15-29              |  4747563 |          0.581 |
| 30 (capped)        |  3070057 |          0.456 |

## Label base rate in the target orders

| split   |   candidates |   positive_rate |
|:--------|-------------:|----------------:|
| test    |      1693252 |          0.0981 |
| train   |      5936277 |          0.0978 |
| valid   |       845132 |          0.0971 |
