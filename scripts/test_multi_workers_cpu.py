# test2.py
import ray
import time  


# ray.init(address='auto')
ray.init('localhost:6379')


@ray.remote#(num_cpus=4)
def isprime(x):  
    if x > 1:  
        for i in range(2, x):  
            if (x % i) == 0:
                return 0  
        else:  
            return x
    return 0


def main():
    lower = 9000000
    upper = 9010000
    primes = []
    objects = []
    start_time = time.time()


    for num in range(lower, upper + 1):
        x=isprime.remote(num)
        objects.append(x)  
   
    objs = ray.get(objects)


    [primes.append(x) for x in objs if x > 0]
    print(len(primes), primes[0], primes[-1])
    print("Time Elapsed:", (time.time() - start_time))  


if __name__ == "__main__":
    main()