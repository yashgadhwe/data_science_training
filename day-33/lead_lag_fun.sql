-- lead function and lag fuction 

use emp;

select * from employees;

-- example of lead fuction and the lag function 
select employeeid, salary, 
lag(salary,3) over() as LG,
lead(salary, 3) over() as LD
from employees;

-- ---------------------------------------------------------------------------------

-- find out whether the current employees have more salary than the previous employees or not 
select * from
(
	select employeeid, salary, 
	lag(salary,3) over() as LG
	from employees
)as g 
where salary > LG;
