-- get the customers data and any one orders of those customers without using join

use classicmodels;

-- use when we don't have the same column from the both table 
-- select customernumber, ordernumber, country from customers inner join orders
-- on customers.customernumber = orders.customernumber; 

select * from customers;

select * from orders;

-- use when the column name is same from both the table 
select customernumber, ordernumber, orderdate, country from customers join orders
using (customernumber);

-- --------------------------------------------------------------------------------------------
-- (aggregation function)
-- avg credit limit by each country
-- grouping

select country, avg(creditlimit)  as ag 
from customers
group by country; 

-- --------------------------------------------------------------------------------------------

-- get the country wies number of orders
-- here we want country , ordernumber column data

select country, count(ordernumber) as order_per_coutry from customers join orders 
using(customernumber)
group by country ;

-- top 3 country which has highest number of orders

select country, count(ordernumber) as order_per_coutry from customers join orders 
using(customernumber)
group by country 
order by order_per_coutry desc limit 3;

-- --------------------------------------------------------------------------------------------

-- get that country which have highest number of customers who have not placed any order
-- here, we want country, customernumber, ordernumber
select country, customernumber, ordernumber 
from customers left join orders
using(customernumber)
where ordernumber is null
order by customernumber desc;

-- get the country who have most number of customer who not placed any order

select country, count(customernumber) as cnt 
from customers left join orders
using(customernumber)
where ordernumber is null
group by country
order by cnt desc;

-- --------------------------------------------------------------------------------------------

-- get the most expensive ordervalue





