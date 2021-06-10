from fruitify.loaders import load_fruit2def


def main():
    fruit2def = load_fruit2def()
    for pair in fruit2def:
        print(pair)
    print("in total:", len(fruit2def))

    
if __name__ == '__main__':
    main()
