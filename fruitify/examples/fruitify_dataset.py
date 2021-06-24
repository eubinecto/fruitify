from fruitify.loaders import load_fruitify_dataset


def main():
    fruit2def = load_fruitify_dataset()
    for pair in fruit2def:
        print(pair)
    print("in total:", len(fruit2def))

    
if __name__ == '__main__':
    main()
