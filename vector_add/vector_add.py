import random
import time

def main():
    N = 1_000_000

    A, B = [0] * N, [0] * N

    for i in range(N):
        A[i] = random.randint(0, 100)
        B[i] = random.randint(0, 100)

    t0 = time.time()
    C = list(map(lambda a, b: a + b, A, B))
    print(f'Python Vector Add Time: {(time.time() - t0) * 1000:.4f} ms')

if __name__ == '__main__':
    main()
