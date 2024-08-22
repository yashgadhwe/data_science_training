
create database emp;  

use emp;
-- Create Employees table
CREATE TABLE Employees (
    EmployeeID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Department VARCHAR(50),
    Position VARCHAR(50),
    Salary DECIMAL(10, 2),
    ManagerID INT
);
 
 -- Insert 25 rows of data into the Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Department, Position, Salary, ManagerID)
VALUES
(1, 'John', 'Doe', 'HR', 'HR Manager', 75000, NULL),
(2, 'Jane', 'Smith', 'HR', 'Recruiter', 50000, 1),
(3, 'Mike', 'Johnson', 'IT', 'IT Manager', 85000, NULL),
(4, 'Sara', 'Davis', 'IT', 'Developer', 60000, 3),
(5, 'Chris', 'Lee', 'IT', 'Developer', 62000, 3),
(6, 'Anna', 'Brown', 'IT', 'System Administrator', 58000, 3),
(7, 'David', 'Wilson', 'Sales', 'Sales Manager', 80000, NULL),
(8, 'Karen', 'Taylor', 'Sales', 'Sales Executive', 55000, 7),
(9, 'James', 'Anderson', 'Sales', 'Sales Executive', 54000, 7),
(10, 'Patricia', 'Thomas', 'Marketing', 'Marketing Manager', 78000, NULL),
(11, 'Linda', 'Jackson', 'Marketing', 'Marketing Specialist', 50000, 10),
(12, 'Robert', 'White', 'Finance', 'Finance Manager', 90000, NULL),
(13, 'Lisa', 'Harris', 'Finance', 'Accountant', 58000, 12),
(14, 'Steven', 'Martin', 'Finance', 'Financial Analyst', 62000, 12),
(15, 'Karen', 'Thompson', 'HR', 'HR Specialist', 53000, 1),
(16, 'Mark', 'Garcia', 'IT', 'Network Engineer', 60000, 3),
(17, 'Nancy', 'Martinez', 'IT', 'Developer', 61000, 3),
(18, 'Charles', 'Robinson', 'Sales', 'Sales Executive', 56000, 7),
(19, 'Jessica', 'Clark', 'Marketing', 'Content Creator', 52000, 10),
(20, 'Matthew', 'Rodriguez', 'Finance', 'Accountant', 59000, 12),
(21, 'Sarah', 'Lewis', 'HR', 'Recruiter', 51000, 1),
(22, 'Paul', 'Lee', 'IT', 'System Analyst', 63000, 3),
(23, 'Jennifer', 'Walker', 'Sales', 'Sales Executive', 57000, 7),
(24, 'Daniel', 'Hall', 'Marketing', 'SEO Specialist', 54000, 10),
(25, 'Laura', 'Allen', 'Finance', 'Accountant', 60000, 12);

-- -----------------------------------------------------------------------------------
use emp;
select * from employees;

-- get the employees who having the same salary 

select distinct a.employeeid, a.salary
from employees as a, employees as b 
where a.salary = b.salary and a.employeeid != b.employeeid
order by a.salary;

-- find out the employees who belongs from same department
select * from employees;

select distinct a.employeeid, a.department
from employees as a, employees as b 
where a.department = b.department and a.employeeid != b.employeeid;

