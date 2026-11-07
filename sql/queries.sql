-- Q1 — WHERE then HAVING
SELECT
    customer_id,
    SUM(order_amount) AS qualifying_spend_inr
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend_inr DESC, customer_id;

-- Q2 — revenue by customer city and product category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3 — products priced above the average catalogue price
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4 — customers with no completed order
SELECT c.customer_id, c.city
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id
 AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5 — customers whose completed spend exceeds the customer average
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
), spend_average AS (
    SELECT AVG(total_spend_inr) AS average_spend_inr
    FROM customer_spend
)
SELECT s.customer_id, s.total_spend_inr, a.average_spend_inr
FROM customer_spend AS s
CROSS JOIN spend_average AS a
WHERE s.total_spend_inr > a.average_spend_inr
ORDER BY s.total_spend_inr DESC, s.customer_id;

-- Q6 — city-level spend ranking with ties preserved
WITH customer_spend AS (
    SELECT
        c.city,
        c.customer_id,
        SUM(o.order_amount) AS total_spend_inr
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT
    city,
    customer_id,
    total_spend_inr,
    RANK() OVER (PARTITION BY city ORDER BY total_spend_inr DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend_inr DESC) AS dense_spend_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend_inr DESC) AS row_num
FROM customer_spend
ORDER BY city, row_num;

-- Q7 — daily completed revenue with a cumulative running total
WITH daily_revenue AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue_inr, 2) AS revenue_inr,
    ROUND(SUM(revenue_inr) OVER (ORDER BY day), 2) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY day;
