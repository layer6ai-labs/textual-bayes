import pickle


def load_from_disk(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_to_disk(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)
