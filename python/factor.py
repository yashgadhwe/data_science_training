# create function to find the factorial of the given number

def myfactor(x):
    for i in range(1, x+1):
        if x%i== 0:
            print(i)

myfactor(20)