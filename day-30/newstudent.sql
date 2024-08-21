-- create database NewStudent
--  apply these below things
-- create student, course, course fees tables of your desired column
-- student can buy the couses which is the part of our course table
-- course fees should vary from 40000 to 80000
-- if students address is not present make default value as hyderabad  

create database NEWSTUDENT;  

create table STUDENT(
	SID int primary key,
    SNAME varchar(30),
    ADDR char(50) default "Hyderabad"
);

create table COURSE(
	CID int primary key,
    CNAME varchar(30)
);

create table COURSEFEES(
	TID int,
    FEES int, check(FEES between 40000 and 80000),
    CID int,
    SID int,
    foreign key(CID) references COURSE(CID),
    foreign key(SID) references STUDENT(SID)
)