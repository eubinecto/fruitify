
def foo(a, b, c):
    return a + b + c


def main():
    kwargs = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4
    }
    return foo(**kwargs)


if __name__ == '__main__':
    main()
