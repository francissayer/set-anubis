import gzip
import pickle
from typing import Dict
import pandas as pd

def save_bundle(bundle: Dict[str, pd.DataFrame], filepath: str) -> None:
    """
    Sauvegarde un dict[str -> DataFrame] dans un seul fichier compressé.
    Exemple: filepath='events.pkl.gz'
    """
    with gzip.open(filepath, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_bundle(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Recharge le dict[str -> DataFrame] exactement comme à l’origine.
    """
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        import io
        with io.FileIO(filepath) as f:
            return pickle.load(f)
    
if __name__ == "__main__":
    dict_load = load_bundle("paul_dict.pkl.gz")
    
    print(dict_load.keys())
    print(dict_load["LLPs"])
    print(dict_load["LLPchildren"])