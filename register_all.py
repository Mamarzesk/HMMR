import os
from registration import register


def main(data_path):
    datasets = [f.name for f in os.scandir(data_path) if f.is_dir()]
    if "BITE" not in datasets:
        raise RuntimeError("BITE dataset not present.")
    if "RESECT" not in datasets:
        raise RuntimeError("RESECT dataset not present.")

    bite_g2 = os.path.join(data_path, "BITE", "group2")
    bite_g2_cases = [
        f.name for f in os.scandir(bite_g2) if f.is_dir() and f.name != "01"
    ]
    bite_paths = {}
    bite_g4 = os.path.join(data_path, "BITE", "group4")
    bite_g4_cases = [
        f.name for f in os.scandir(bite_g4) if f.is_dir() and f.name != "01"
    ]
    resect = os.path.join(data_path, "RESECT", "MINC")
    resect_cases = [f.name for f in os.scandir(resect) if f.is_dir()]
    resect_paths = {}


def run_group_experiments(path, cases):
    for case in cases:
        case_path = os.path.join(path, case)


if __name__ == "__main__":
    data_path = os.path.abspath("/data")
    main(data_path)
