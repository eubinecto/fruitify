from typing import List, Tuple
from fruitify.paths import FRUIT2DEFS_TSV
import csv


def load_fruit2def() -> List[Tuple[str, str]]:
    with open(FRUIT2DEFS_TSV, 'r') as fh:
        csv_reader = csv.reader(fh, delimiter="\t")
        # skip the header
        next(csv_reader)
        fruit2def = list()
        for row in csv_reader:
            fruit2def += [
                (row[0].strip(), definition.strip())
                for definition in row[1:9]
            ]
        return fruit2def
