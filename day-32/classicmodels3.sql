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
select country, avg(shippeddate - orderdate) as delay
from orders join customers
using(customernumber)



 



