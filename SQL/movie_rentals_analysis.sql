USE movie_rental;
-- most popular films

SELECT 
f.film_id,
title,
COUNT(rental_id) AS #rentals
FROM
rental r
JOIN inventory i
ON r.inventory_id = i.inventory_id
JOIN film f
On f.film_id = i.film_id
GROUP BY f.film_id, f.title
ORDER BY #rentals DESC;

-- most popular categories

SELECT
	fc.category_id,
	c.name AS category_name,
	COUNT(r.rental_id) AS #rentals

FROM
	film_category fc
JOIN category c
ON fc.category_id = c.category_id
JOIN film f
On fc.film_id = f.film_id
JOIN inventory i
ON f.film_id = i.film_id
JOIN rental r
ON i.inventory_id = r.inventory_id
GROUP BY fc.category_id, c.name
ORDER BY #rentals DESC;

-- Who are the highest-value customers?


SELECT
	c.customer_id,
	c.first_name+ ' ' + c.last_name AS customer_name,
	ROUND(SUM(p.amount),2) AS total_payments
FROM
	customer c
JOIN
	payment p
ON c.customer_id = p.customer_id
GROUP BY c.customer_id, c.first_name+ ' ' + c.last_name
ORDER BY total_payments DESC;

-- What is the revenue generated per each film category

SELECT 
c.category_id,
c.name AS cateogory,
ROUND(SUM(p.amount),2) AS revenue
FROM
	payment p
JOIN rental r
ON p.rental_id = r.rental_id
JOIN inventory i
ON r.inventory_id = i.inventory_id
JOIN film f
ON i.film_id = f.film_id
JOIN film_category fc
ON f.film_id = fc.film_id
JOIN category c
ON fc.category_id = c.category_id
GROUP BY c.category_id, c.name
ORDER BY revenue DESC;

-- Which store has generated most revenue

SELECT
s.store_id,
ROUND(SUM(amount),2) AS revenue
FROM
	store s
JOIN staff st
On s.store_id = st.store_id
JOIN payment p
On p.staff_id = st.staff_id
GROUP BY s.store_id;

-- One-time vs repeat customers
SELECT
	SUM(CASE WHEN #purchase > 1 THEN 1 ELSE 0 END) AS repeated,
	SUM(CASE WHEN #purchase > 1 THEN 0 ELSE 1 END) AS one_timer
FROM
	(SELECT
		customer_id,
		COUNT(payment_id) AS #purchase
	FROM
		payment
	GROUP BY customer_id)q;

-- identifying inventory items that are not frequently rented

SELECT
	f.film_id,
	f.title,
	c.name AS category,
	COUNT(rental_id) AS #rentals
FROM
	rental r
JOIN inventory i
ON r.inventory_id = i.inventory_id
JOIN film f
On i.film_id = f.film_id
JOIN film_category fc
On f.film_id = fc.film_id
JOIN category c
ON fc.category_id = c.category_id
GROUP BY f.film_id, title, c.name
ORDER BY #rentals;

-- employee contribution

SELECT
	s.staff_id,
	first_name + ' ' + last_name AS staff_name,
	COUNT(p.payment_id) AS total_processed_payments
FROM
	staff s
JOIN 
	payment p
ON s.staff_id = p.staff_id
GROUP BY s.staff_id, first_name + ' ' + last_name;

-- Customer retention frequency

WITH rental_gaps AS(
	SELECT 
		customer_id,
		rental_date,
		LAG(rental_date) OVER(PARTITION BY customer_id ORDER BY rental_date) AS previous_date
	FROM rental
)

SELECT
	customer_id,
	COUNT(customer_id) AS #rentals,
	AVG(DATEDIFF(day, previous_date, rental_date)) AS avg_frequency_in_days

FROM 
	rental_gaps
GROUP BY customer_id
ORDER BY #rentals DESC, avg_frequency_in_days DESC;

-- revenue by city

SELECT
	city,
	ROUND(SUM(amount),2) AS revenue
FROM 
	payment p
JOIN customer ct
ON p.customer_id = ct.customer_id
JOIN address a
ON ct.address_id = a.address_id
JOIN city c
ON a.city_id = c.city_id
GROUP BY city
ORDER BY revenue DESC;

-- revenue by country

SELECT
	country,
	ROUND(SUM(amount),2) AS revenue
FROM 
	payment p
JOIN customer ct
ON p.customer_id = ct.customer_id
JOIN address a
ON ct.address_id = a.address_id
JOIN city c
ON a.city_id = c.city_id
JOIN country co
ON co.country_id = c.country_id
GROUP BY country
ORDER BY revenue DESC;

-- Monthly payment trend

SELECT
	MONTH(payment_date) AS month,
	ROUND(SUM(amount),2) AS revenue
FROM
	payment
GROUP BY 
	MONTH(payment_date)
ORDER BY revenue DESC;

-- Monthly rental trend

SELECT 
	FORMAT(rental_date, 'yyyy-MM') AS rental_month,
	COUNT(rental_id) AS #rentals
FROM
	rental
GROUP BY FORMAT(rental_date, 'yyyy-MM')
ORDER BY rental_month DESC;

-- RFM analysis
DECLARE @as_of_date DATE = '2006-02-15';

WITH RFM AS (
    SELECT 
        r.customer_id,
        COUNT(r.customer_id) AS frequency,
        DATEDIFF(DAY, MAX(rental_date), @as_of_date) AS recency,
        ROUND(SUM(amount), 2) AS monetary_value
    FROM rental r
    JOIN payment p ON r.customer_id = p.customer_id
    GROUP BY r.customer_id
),

rfm_ranked AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency ASC) AS R_score,
        NTILE(5) OVER (ORDER BY frequency DESC) AS F_score,
        NTILE(5) OVER (ORDER BY monetary_value DESC) AS M_score
    FROM RFM
),

rfm_final AS (
    SELECT *,
        R_score + F_score + M_score AS RFM_Score
    FROM rfm_ranked
)

SELECT *,
    CASE 
        WHEN RFM_Score >= 13 THEN 'Platinum'
        WHEN RFM_Score >= 10 THEN 'Gold'
        WHEN RFM_Score >= 7 THEN 'Silver'
        ELSE 'Bronze'
    END AS rfm_segment
FROM rfm_final;
