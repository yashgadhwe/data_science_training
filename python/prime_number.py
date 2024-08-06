def prime(x):
    cnt = 0
    x = x
    for i in range(1,x+1):
        if x%i== 0:
            cnt = cnt +1

    if cnt == 2:
        print("prime")
    else:
        print("not prime")

prime(5)
