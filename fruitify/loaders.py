from typing import List, Tuple
from fruitify.paths import FRUITIFY_DATASET_TSV
import csv


def load_fruitify_dataset() -> List[List[str]]:
    """
    (en, kr, lang, def)
    :return:
    """
    with open(FRUITIFY_DATASET_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        # skip the header
        next(tsv_reader)
        return [
            row
            for row in tsv_reader
        ]
