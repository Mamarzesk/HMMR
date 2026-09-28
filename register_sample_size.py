import os
import time
import pandas as pd
from numpy import mean
from plot_results import plot_results
from registration import register
from registration_nonlinear import register_nonlinear


def main(data_path, skip_compute=False):
    output_root = os.path.join(data_path, "sample_size")
    rigid_root = os.path.join(data_path, "rigid_paper")
    datasets = [f.name for f in os.scandir(data_path) if f.is_dir()]
    if "BITE" not in datasets:
        raise RuntimeError("BITE dataset not present.")
    if "RESECT" not in datasets:
        raise RuntimeError("RESECT dataset not present.")

    bite_g2 = os.path.join(data_path, "BITE", "group2")
    bite_g2_out = os.path.join(output_root, "bite_group2")
    bite_g2_rigid = os.path.join(rigid_root, "bite_group2")
    bite_g2_cases = [
        f.name
        for f in os.scandir(bite_g2)
        if f.is_dir() and f.name not in ["01", "recon", ".mypy_cache"]
    ]
    bite_g2_paths = {
        "fixed": "ReconUS.mnc",
        "moving": "mr.mnc",
        "tag": "_all.tag",
        "add_case": True,
    }
    bite_g4 = os.path.join(data_path, "BITE", "group4")
    bite_g4_out = os.path.join(output_root, "bite_group4")
    bite_g4_rigid = os.path.join(rigid_root, "bite_group4")
    bite_g4_cases = [
        f.name for f in os.scandir(bite_g4) if f.is_dir() and f.name != "14"
    ]
    bite_g4_paths = {
        "fixed": "US3DT.mnc",
        "moving": "MR.mnc",
        "tag": "Tags.tag",
        "folder": "3D",
        "add_case": False,
    }
    resect = os.path.join(data_path, "RESECT", "MINC")
    resect_out = os.path.join(output_root, "resect")
    resect_rigid = os.path.join(rigid_root, "resect")
    resect_pre_cases = [
        f.name for f in os.scandir(resect) if f.is_dir() and f.name not in ("Case11")
    ]
    resect_pre_cases.append("Case1")
    resect_post_cases = [
        f.name
        for f in os.scandir(resect)
        if f.is_dir() and f.name not in ("Case5", "Case13", "Case26")
    ]
    resect_pre_flair_paths = {
        "fixed": "US/US_test.mnc",
        "moving": "MRI/FLAIR_test.mnc",
        "tag_folder": "landmarks",
        "tag": "-MRI-beforeUS.tag",
        "add_case": True,
    }
    resect_pre_flair_out = os.path.join(resect_out, "pre", "flair")
    resect_pre_flair_rigid = os.path.join(resect_rigid, "pre", "flair")
    resect_post_flair_paths = {
        "fixed": "US/US_post_test.mnc",
        "moving": "MRI/FLAIR_test.mnc",
        "tag_folder": "landmarks",
        "tag": "-MRI-afterUS.tag",
        "add_case": True,
    }
    resect_post_flair_out = os.path.join(resect_out, "post", "flair")
    resect_post_flair_rigid = os.path.join(resect_rigid, "post", "flair")
    resect_pre_t1_paths = {
        "fixed": "US/US_test.mnc",
        "moving": "MRI/T1_test_reg.mnc",
        "tag_folder": "landmarks",
        "tag": "-MRI-beforeUS.tag",
        "add_case": True,
    }
    resect_pre_t1_out = os.path.join(resect_out, "pre", "t1")
    resect_pre_t1_rigid = os.path.join(resect_rigid, "pre", "t1")
    resect_post_t1_paths = {
        "fixed": "US/US_post_test.mnc",
        "moving": "MRI/T1_test_reg.mnc",
        "tag_folder": "landmarks",
        "tag": "-MRI-afterUS.tag",
        "add_case": True,
    }
    resect_post_t1_out = os.path.join(resect_out, "post", "t1")
    resect_post_t1_rigid = os.path.join(resect_rigid, "post", "t1")
    group_paths = [bite_g2, bite_g4, resect, resect, resect, resect]
    cases_list = [
        bite_g2_cases,
        bite_g4_cases,
        resect_pre_cases,
        resect_post_cases,
        resect_pre_cases,
        resect_post_cases,
    ]
    in_paths_list = [
        bite_g2_paths,
        bite_g4_paths,
        resect_pre_flair_paths,
        resect_post_flair_paths,
        resect_pre_t1_paths,
        resect_post_t1_paths,
    ]
    out_list = [
        bite_g2_out,
        bite_g4_out,
        resect_pre_flair_out,
        resect_post_flair_out,
        resect_pre_t1_out,
        resect_post_t1_out,
    ]
    rigid_list = [
        bite_g2_rigid,
        bite_g4_rigid,
        resect_pre_flair_rigid,
        resect_post_flair_rigid,
        resect_pre_t1_rigid,
        resect_post_t1_rigid,
    ]

    # affine_times_5k = []
    # affine_times_10k = []
    # affine_times_20k = []
    # nonlinear_times_5k = []
    # nonlinear_times_10k = []
    # nonlinear_times_20k = []

    affine_times_50k = []
    affine_times_100k = []
    nonlinear_times_50k = []
    nonlinear_times_100k = []
    for group_path, case_list, in_paths, out, rigid in zip(
        group_paths, cases_list, in_paths_list, out_list, rigid_list
    ):
        affine_times, nonlinear_times = run_group_experiments(
            group_path, case_list, in_paths, out, rigid, skip_compute=skip_compute
        )
        # affine_times_5k.extend(affine_times[0])
        # affine_times_10k.extend(affine_times[1])
        # affine_times_20k.extend(affine_times[2])
        # nonlinear_times_5k.extend(nonlinear_times[0])
        # nonlinear_times_10k.extend(nonlinear_times[1])
        # nonlinear_times_20k.extend(nonlinear_times[2])
        affine_times_50k.extend(affine_times[0])
        affine_times_100k.extend(affine_times[1])
        nonlinear_times_50k.extend(nonlinear_times[0])
        nonlinear_times_100k.extend(nonlinear_times[1])

    registration_types = [
        "affine",
        "affine",
        # "affine",
        "nonlinear",
        "nonlinear",
        # "nonlinear",
    ]
    # sample_sizes = ["5k", "10k", "20k", "5k", "10k", "20k"]
    sample_sizes = ["50k", "100k", "50k", "100k"]
    # mean_times = [
    #     mean(affine_times_5k),
    #     mean(affine_times_10k),
    #     mean(affine_times_20k),
    #     mean(nonlinear_times_5k),
    #     mean(nonlinear_times_10k),
    #     mean(nonlinear_times_20k),
    # ]
    mean_times = [
        mean(affine_times_50k),
        mean(affine_times_100k),
        mean(nonlinear_times_50k),
        mean(nonlinear_times_100k),
    ]
    df = pd.DataFrame(
        {
            "registration type": registration_types,
            "sample size": sample_sizes,
            "average registration time": mean_times,
        }
    )

    print(df)
    df.to_csv(os.path.join(output_root, "statistics.csv"), sep=",", index=False)


def run_group_experiments(group_path, cases, in_paths, out, rigid_folder, skip_compute=False):
    affine_times = []
    nonlinear_times = []
    sigma_folder = "sigma_1.0"
    bspline_spacing = 20.0
    dynamic_sampling = True
    affine_folder = "affine"
    nonlinear_folder = "non_linear"

    # for sample_size in [5000, 10_000, 20_000]:
    for sample_size in [50_000, 100_000]:
        sample_folder = f"sample_size_{str(sample_size)}"
        affine_sample_output_folder = os.path.join(out, affine_folder, sample_folder)
        nonlinear_sample_output_folder = os.path.join(out, nonlinear_folder, sample_folder)
        os.makedirs(affine_sample_output_folder, exist_ok=True)
        os.makedirs(nonlinear_sample_output_folder, exist_ok=True)
        for case in cases:
            if skip_compute:
                continue
            case_path = os.path.join(group_path, case)
            affine_case_output_path = os.path.join(affine_sample_output_folder, case)
            os.makedirs(affine_case_output_path, exist_ok=True)
            nonlinear_case_output_path = os.path.join(nonlinear_sample_output_folder, case)
            os.makedirs(nonlinear_case_output_path, exist_ok=True)
            if "folder" in in_paths:
                case_path = os.path.join(case_path, in_paths["folder"])
            fixed = os.path.join(case_path, in_paths["fixed"])
            moving = os.path.join(case_path, in_paths["moving"])
            rigid_transformation = os.path.join(
                rigid_folder, "rigid", sigma_folder, case, "affine_transformations.csv"
            )
            tag = case_path
            tag_file = in_paths["tag"]
            if "tag_folder" in in_paths:
                tag = os.path.join(tag, in_paths["tag_folder"])
            if "add_case" in in_paths:
                if in_paths["add_case"]:
                    tag_file = f"{case}{tag_file}"
            tag = os.path.join(tag, tag_file)
            times = []
            # for fold in range(5):
            for fold in range(1):
                # affine_fold_output_path = os.path.join(affine_case_output_path, f"fold_{fold}")
                # os.makedirs(affine_fold_output_path, exist_ok=True)
                start = time.perf_counter()
                register(
                    fixed,
                    moving,
                    tag,
                    affine_case_output_path,
                    sigma=1.0,
                    force_rigid=False,
                    samples_count=sample_size,
                )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            affine_times.append(times)
            times = []
            # for fold in range(5):
            for fold in range(1):
                # nonlinear_fold_output_path = os.path.join(nonlinear_case_output_path, f"fold_{fold}")
                # os.makedirs(nonlinear_fold_output_path, exist_ok=True)
                start = time.perf_counter()
                register_nonlinear(
                    fixed,
                    moving,
                    tag,
                    nonlinear_case_output_path,
                    sigma=1.0,
                    bspline_spacing=bspline_spacing,
                    rigid_file=rigid_transformation,
                    dynamic_sampling=dynamic_sampling,
                    force_rigid=True,
                )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            nonlinear_times.append(times)
                
        plot_results(
            affine_sample_output_folder, f"affine transformation, sample size = {sample_size}"
        )
        plot_results(
            nonlinear_sample_output_folder, f"nonlinear transformation, sample size = {sample_size}", nonlinear=True)
    return affine_times, nonlinear_times


if __name__ == "__main__":
    data_path = os.path.abspath("/data")
    main(data_path)
