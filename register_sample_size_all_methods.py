import os
import time
import pandas as pd
from numpy import mean
from plot_results import plot_results
from affine_GI import affine_gi
from affine_GOA import affine_goa
from affine_LC2 import affine_lc2
from nonlinear_GI import nonlinear_gi
from nonlinear_GOA import nonlinear_goa
from nonlinear_LC2 import nonlinear_lc2


def main(data_path, skip_compute=False):
    output_root = os.path.join(data_path, "sample_size_study")
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

    affine_gi_times_5k = []
    affine_gi_times_10k = []
    affine_gi_times_20k = []
    affine_goa_times_5k = []
    affine_goa_times_10k = []
    affine_goa_times_20k = []
    affine_lc2_times_5k = []
    affine_lc2_times_10k = []
    affine_lc2_times_20k = []
    nl_gi_times_5k = []
    nl_gi_times_10k = []
    nl_gi_times_20k = []
    nl_goa_times_5k = []
    nl_goa_times_10k = []
    nl_goa_times_20k = []
    nl_lc2_times_5k = []
    nl_lc2_times_10k = []
    nl_lc2_times_20k = []

    for group_path, case_list, in_paths, out in zip(
        group_paths, cases_list, in_paths_list, out_list
    ):
        (
            affine_gi_times,
            affine_goa_times,
            affine_lc2_times,
            nl_gi_times,
            nl_goa_times,
            nl_lc2_times,
        ) = run_group_experiments(
            group_path, case_list, in_paths, out, skip_compute=skip_compute
        )
        affine_gi_times_5k.extend(affine_gi_times[0])
        affine_gi_times_10k.extend(affine_gi_times[1])
        affine_gi_times_20k.extend(affine_gi_times[2])
        affine_goa_times_5k.extend(affine_goa_times[0])
        affine_goa_times_10k.extend(affine_goa_times[1])
        affine_goa_times_20k.extend(affine_goa_times[2])
        affine_lc2_times_5k.extend(affine_lc2_times[0])
        affine_lc2_times_10k.extend(affine_lc2_times[1])
        affine_lc2_times_20k.extend(affine_lc2_times[2])
        nl_gi_times_5k.extend(nl_gi_times[0])
        nl_gi_times_10k.extend(nl_gi_times[1])
        nl_gi_times_20k.extend(nl_gi_times[2])
        nl_goa_times_5k.extend(nl_goa_times[0])
        nl_goa_times_10k.extend(nl_goa_times[1])
        nl_goa_times_20k.extend(nl_goa_times[2])
        nl_lc2_times_5k.extend(nl_lc2_times[0])
        nl_lc2_times_10k.extend(nl_lc2_times[1])
        nl_lc2_times_20k.extend(nl_lc2_times[2])

    algs = [
        "affine gi",
        "affine gi",
        "affine gi",
        "affine goa",
        "affine goa",
        "affine goa",
        "affine lc2",
        "affine lc2",
        "affine lc2",
        "nl gi",
        "nl gi",
        "nl gi",
        "nl goa",
        "nl goa",
        "nl goa",
        "nl lc2",
        "nl lc2",
        "nl lc2",
    ]
    sample_sizes = ["5k", "10k", "20k"] * 6
    mean_times = [
        mean(affine_gi_times_5k),
        mean(affine_gi_times_10k),
        mean(affine_gi_times_20k),
        mean(affine_goa_times_5k),
        mean(affine_goa_times_10k),
        mean(affine_goa_times_20k),
        mean(affine_lc2_times_5k),
        mean(affine_lc2_times_10k),
        mean(affine_lc2_times_20k),
        mean(nl_gi_times_5k),
        mean(nl_gi_times_10k),
        mean(nl_gi_times_20k),
        mean(nl_goa_times_5k),
        mean(nl_goa_times_10k),
        mean(nl_goa_times_20k),
        mean(nl_lc2_times_5k),
        mean(nl_lc2_times_10k),
        mean(nl_lc2_times_20k),
    ]
    df = pd.DataFrame({"algorithm": algs, "sample size": sample_sizes, "average registration time": mean_times})

    print(df)
    df.to_csv(os.path.join(output_root, "statistics.csv"), sep=",", index=False)


def run_group_experiments(group_path, cases, in_paths, out, rigid="rigid_paper", skip_compute=False):
    affine_gi_times = []
    affine_goa_times = []
    affine_lc2_times = []

    nonlinear_gi_times = []
    nonlinear_goa_times = []
    nonlinear_lc2_times = []

    bspline_spacing = 20.0
    dynamic_sampling = True
    force_rigid = True

    for folder, register, times in zip(
        ["affine_gi", "affine_goa", "affine_lc2"],
        [affine_gi, affine_goa, affine_lc2],
        [affine_gi_times, affine_goa_times, affine_lc2_times],
    ):
        for sample_size in [5000, 10_000, 20_000]:
            sample_folder = f"sample_size_{sample_size}"
            sample_output_folder = os.path.join(out, folder, sample_folder)
            os.makedirs(sample_output_folder, exist_ok=True)
            for case in cases:
                if skip_compute:
                    continue
                case_path = os.path.join(group_path, case)
                case_output_path = os.path.join(sample_output_folder, case)
                os.makedirs(case_output_path, exist_ok=True)
                if "folder" in in_paths:
                    case_path = os.path.join(case_path, in_paths["folder"])
                fixed = os.path.join(case_path, in_paths["fixed"])
                moving = os.path.join(case_path, in_paths["moving"])
                tag = case_path
                tag_file = in_paths["tag"]
                if "tag_folder" in in_paths:
                    tag = os.path.join(tag, in_paths["tag_folder"])
                if "add_case" in in_paths:
                    if in_paths["add_case"]:
                        tag_file = f"{case}{tag_file}"
                tag = os.path.join(tag, tag_file)
                sample_times = []
                for _ in range(5):
                    start = time.perf_counter()
                    register(
                        fixed,
                        moving,
                        tag,
                        case_output_path,
                        sigma=1,
                        force_rigid=force_rigid,
                        samples_count=sample_size,
                    )
                    elapsed = time.perf_counter() - start
                    sample_times.append(elapsed)
                times.append(sample_times)
            plot_results(
                sample_output_folder,
                f"{folder} transformation, sample size {sample_size}",
            )
    for folder, affine_folder, register, times in zip(
        ["nonlinear_gi", "nonlinear_goa", "nonlinear_lc2"],
        ["affine_gi", "affine_goa", "affine_lc2"],
        [nonlinear_gi, nonlinear_goa, nonlinear_lc2],
        [nonlinear_gi_times, nonlinear_goa_times, nonlinear_lc2_times],
    ):
        for sample_size in [5000, 10_000, 20_000]:
            sample_folder = f"sample_size_{sample_size}"
            sample_output_folder = os.path.join(out, folder, sample_folder)
            os.makedirs(sample_output_folder, exist_ok=True)
            for case in cases:
                if skip_compute:
                    continue
                case_path = os.path.join(group_path, case)
                case_output_path = os.path.join(sample_output_folder, case)
                affine_matrix = (
                    os.path.join(  # TODO: make sure the one in paper is used
                        rigid, affine_folder, sample_folder, case, "transformation.csv"
                    )
                )
                os.makedirs(case_output_path, exist_ok=True)
                if "folder" in in_paths:
                    case_path = os.path.join(case_path, in_paths["folder"])
                fixed = os.path.join(case_path, in_paths["fixed"])
                moving = os.path.join(case_path, in_paths["moving"])
                tag = case_path
                tag_file = in_paths["tag"]
                if "tag_folder" in in_paths:
                    tag = os.path.join(tag, in_paths["tag_folder"])
                if "add_case" in in_paths:
                    if in_paths["add_case"]:
                        tag_file = f"{case}{tag_file}"
                tag = os.path.join(tag, tag_file)
                sample_times = []
                for _ in range(5):
                    start = time.perf_counter()
                    register(
                        fixed,
                        moving,
                        tag,
                        affine_matrix,
                        case_output_path,
                        sigma=1,
                        bspline_spacing=bspline_spacing,
                        force_rigid=force_rigid,
                        dynamic_sampling=dynamic_sampling,
                        samples_count=sample_size,
                    )
                    elapsed = time.perf_counter() - start
                    sample_times.append(elapsed)
                times.append(sample_times)
            plot_results(
                sample_output_folder,
                f"{folder} transformation, sample size = {sample_size}",
                nonlinear=True,
            )

    return (
        affine_gi_times,
        affine_goa_times,
        affine_lc2_times,
        nonlinear_gi_times,
        nonlinear_goa_times,
        nonlinear_lc2_times,
    )


if __name__ == "__main__":
    data_path = os.path.abspath("/data")
    main(data_path)
