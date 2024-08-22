-- find customerwise each ordernumber order value and sort the data such like that we get highest 
-- ordervalue of each customer on top 

-- get the data - customernumber, ordernumber, order

select customernumber, ordernumber, sum(priceeach*quantityordered) as ordval
from orders join orderdetails 
using (ordernumber)
group by customerNumber, orderNumber 
order by customerNumber asc , ordval desc;

-- ---------------------------------------------------------------------------

-- find out each productline highest sold product (in term of total sales of that product on top)

select productline, productcode, sum(priceeach*quantityordered) as saleval
from products join orderdetails
using(productcode)
group by productLine, productcode 
order by productline asc, saleval desc;

-- ------------------------------------------------------------------------------------------------

-- find out which customer have highest number of orders from each country
-- (sort all the customerdata based upon number of orders from each country)
-- data needed - ordernumber, customernumber, country
select country, customernumber, count(ordernumber) as cnt 
from customers join orders
using(customernumber) 
group by country, customerNumber
order by country asc , cnt desc;

-- ---------------------------------------------------------------------------------------------------

-- queries based on the sql inbuilt functions(like date, string)
-- year wies quaterly sales and sort the data by sales for each year

SELECT 
    YEAR(o.orderDate) AS year,
    QUARTER(o.orderDate) AS quarter,
    SUM(od.quantityOrdered * od.priceEach) AS total_sales
FROM
    orders o
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
GROUP BY 
    year, quarter
ORDER BY 
    year ASC, total_sales DESC;

-- or---------

select year(orderdate) as year, quarter(orderdate) as qt, sum(priceeach * quantityordered) as sales
from orders join orderdetails
using(ordernumber) 
group by year, qt 
order by year asc, sales desc;

-- ------------------------------------------------------------------------------------------------

-- which country have lowest delay in shipping

-- select * from orders;
use classicmodels;
select country, avg(shippeddate - orderdate) as delay
from orders join customers
using(customernumber)
group by country 
order by delay asc;

-- find country wies sales which is having 3 to 5 characters in country name

select country, sum(priceeach * quantityordered) as sales
from customers join orders
using(customernumber) join orderdetails using(ordernumber)
group by country
having length(country) between 3 and 5 ;

-- ------------------------------------------------------------------------------------

-- find out less profitable product from each month

use classicmodels;
-- select * from orderdetails;

SELECT 
    year,
    month,
    productName,
    profit
FROM (
    SELECT 
        YEAR(o.orderDate) AS year,
        MONTH(o.orderDate) AS month,
        p.productName,
        SUM((od.quantityOrdered * od.priceEach) - (od.quantityOrdered * p.buyPrice)) AS profit,
        ROW_NUMBER() OVER (PARTITION BY YEAR(o.orderDate), MONTH(o.orderDate) ORDER BY SUM((od.quantityOrdered * od.priceEach) - (od.quantityOrdered * p.buyPrice)) ASC) AS rn
    FROM
        orders o
        JOIN orderdetails od ON o.orderNumber = od.orderNumber
        JOIN products p ON od.productCode = p.productCode
    GROUP BY 
        year, month, p.productName
) AS ranked_products
WHERE rn = 1
ORDER BY 
    year ASC, month ASC;
    
-- --------------------------------------------------------------------------------------
--  find out the more profitable product from each month,
SELECT 
    year,
    month,
    productName,
    profit
FROM (
    SELECT 
        YEAR(o.orderDate) AS year,
        MONTH(o.orderDate) AS month,
        p.productName,
        SUM((od.quantityOrdered * od.priceEach) - (od.quantityOrdered * p.buyPrice)) AS profit,
        ROW_NUMBER() OVER (PARTITION BY YEAR(o.orderDate), MONTH(o.orderDate) ORDER BY SUM((od.quantityOrdered * od.priceEach) - (od.quantityOrdered * p.buyPrice)) DESC) AS rn
    FROM
        orders o
        JOIN orderdetails od ON o.orderNumber = od.orderNumber
        JOIN products p ON od.productCode = p.productCode
    GROUP BY 
        year, month, p.productName
) AS ranked_products
WHERE rn = 1
ORDER BY 
    year ASC, month ASC;

-- --------------------------------------------------------------------------------------
-- which country is having highest sales of each productline




 



