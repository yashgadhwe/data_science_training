-- day-33 - RowNumber function, Ranking function, Dense Ranking function, analytical function 

use classicmodels;

select customernumber, customername, country, creditlimit,
row_number() over(partition by country order by creditlimit desc) as rownum,
rank() over(partition by country order by creditlimit desc) as rk,
dense_rank() over(partition by country order by creditlimit desc) as drk
from customers;

-- find out top2 creaditlimit from each country

SELECT 
    c1.country,
    c1.customerName,
    c1.creditLimit
FROM 
    customers c1
WHERE 
    (
        SELECT COUNT(DISTINCT c2.creditLimit)
        FROM customers c2
        WHERE c2.country = c1.country AND c2.creditLimit > c1.creditLimit
    ) < 2
ORDER BY 
    c1.country ASC, c1.creditLimit DESC;

-- with using rownumber
SELECT 
    country,
    customerName,
    creditLimit
FROM (
    SELECT 
        c.country,
        c.customerName,
        c.creditLimit,
        ROW_NUMBER() OVER (PARTITION BY c.country ORDER BY c.creditLimit DESC) AS rn
    FROM
        customers c
) AS ranked_customers
WHERE rn <= 2
ORDER BY 
    country ASC, creditLimit DESC;
    
-- or ---

select * from 
(
select country, creditlimit,
row_number() over(partition by country order by creditlimit desc) as RNum
from customers
) as K 
where RNum<=2;

-- ----------------------------------------------------------------------------------------

-- fetch even no.of rows of data
select * from
(
	select *, row_number() over() as RN
	from customers
)as k
where RN%2=0;

-- fetch top 2 customers from each country based upon no.of orders
select * 
from(
	select *, row_number() over(partition by country order by cnt desc) as RN 
	from(
			select country, customernumber, count(ordernumber) as cnt 
			from customers join orders
			using(customernumber) 
			group by country, customerNumber 
		) as k
	)as p
where RN<=2;

-- ----------------------------------------------------------------------------------------
-- find out 2 least profitable product from each country

SELECT 
    country,
    productName,
    profit
FROM (
    SELECT 
        c.country,
        p.productName,
        SUM((od.quantityOrdered * od.priceEach) - (p.buyPrice * od.quantityOrdered)) AS profit,
        ROW_NUMBER() OVER (PARTITION BY c.country ORDER BY SUM((od.quantityOrdered * od.priceEach) - (p.buyPrice * od.quantityOrdered)) ASC) AS rn
    FROM
        customers c
        JOIN orders o ON c.customerNumber = o.customerNumber
        JOIN orderdetails od ON o.orderNumber = od.orderNumber
        JOIN products p ON od.productCode = p.productCode
    GROUP BY 
        c.country, p.productName
) AS ranked_products
WHERE rn <= 2
ORDER BY 
    country ASC, profit ASC;

-- ----------------------------------------------------------------------------------

-- we are getting less profit into certain products analyze why we are making lesser profits

SELECT 
    p.productName,
    p.productLine,
    SUM(od.quantityOrdered) AS total_units_sold,
    SUM(od.quantityOrdered * od.priceEach) AS total_sales,
    SUM(od.quantityOrdered * p.buyPrice) AS total_cost,
    SUM((od.quantityOrdered * od.priceEach) - (od.quantityOrdered * p.buyPrice)) AS total_profit,
    ROUND(SUM((od.quantityOrdered * od.priceEach) - (od.quantityOrdered * p.buyPrice)) / SUM(od.quantityOrdered * od.priceEach) * 100, 2) AS profit_margin_percentage
FROM
    products p
    JOIN orderdetails od ON p.productCode = od.productCode
    JOIN orders o ON od.orderNumber = o.orderNumber
GROUP BY 
    p.productName, p.productLine
ORDER BY 
    total_profit ASC
LIMIT 10; -- or you can remove the limit to see all products

-- --------------------------------------------------------------------------------------

-- find out timeline when we are generating more revenu

SELECT 
    YEAR(o.orderDate) AS year,
    MONTH(o.orderDate) AS month,
    SUM(od.quantityOrdered * od.priceEach) AS total_revenue
FROM
    orders o
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
GROUP BY 
    YEAR(o.orderDate), MONTH(o.orderDate)
ORDER BY 
    total_revenue DESC;
