import os
import time
from numpy import mean
from plot_results import plot_results
from affine_GI import affine_gi
from affine_GOA import affine_goa
from affine_LC2 import affine_lc2
from nonlinear_GI import nonlinear_gi
from nonlinear_GOA import nonlinear_goa
from nonlinear_LC2 import nonlinear_lc2


def main(data_path, skip_compute=False):
    output_root = os.path.join(data_path, "output_all_methods")
    datasets = [f.name for f in os.scandir(data_path) if f.is_dir()]
    if "BITE" not in datasets:
        raise RuntimeError("BITE dataset not present.")
    if "RESECT" not in datasets:
        raise RuntimeError("RESECT dataset not present.")

    bite_g2 = os.path.join(data_path, "BITE", "group2")
    bite_g2_out = os.path.join(output_root, "bite_group2")
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
    resect_post_flair_paths = {
        "fixed": "US/US_post_test.mnc",
        "moving": "MRI/FLAIR_test.mnc",
        "tag_folder": "landmarks",
        "tag": "-MRI-afterUS.tag",
        "add_case": True,
    }
    resect_post_flair_out = os.path.join(resect_out, "post", "flair")
    resect_pre_t1_paths = {
        "fixed": "US/US_test.mnc",
        "moving": "MRI/T1_test_reg.mnc",
        "tag_folder": "landmarks",
        "tag": "-MRI-beforeUS.tag",
        "add_case": True,
    }
    resect_pre_t1_out = os.path.join(resect_out, "pre", "t1")
    resect_post_t1_paths = {
        "fixed": "US/US_post_test.mnc",
        "moving": "MRI/T1_test_reg.mnc",
        "tag_folder": "landmarks",
        "tag": "-MRI-afterUS.tag",
        "add_case": True,
    }
    resect_post_t1_out = os.path.join(resect_out, "post", "t1")
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

    affine_gi_times_cumul = []
    affine_goa_times_cumul = []
    affine_lc2_times_cumul = []
    nl_gi_times_cumul = []
    nl_goa_times_cumul = []
    nl_lc2_times_cumul = []

    for group_path, case_list, in_paths, out in zip(
        group_paths, cases_list, in_paths_list, out_list
    ):
        affine_gi_times, affine_goa_times, affine_lc2_times, nl_gi_times, nl_goa_times, nl_lc2_times = run_group_experiments(
            group_path, case_list, in_paths, out, skip_compute=skip_compute
        )
        affine_gi_times_cumul.extend(affine_gi_times)
        affine_goa_times_cumul.extend(affine_goa_times)
        affine_lc2_times_cumul.extend(affine_lc2_times)
        nl_gi_times_cumul.extend(nl_gi_times)
        nl_goa_times_cumul.extend(nl_goa_times)
        nl_lc2_times_cumul.extend(nl_lc2_times)
    print(
        f"average times for registration: affine gi {mean(affine_gi_times_cumul)} s, affine goa {mean(affine_goa_times_cumul)} s, affine lc2 {mean(affine_lc2_times_cumul)} s, nl gi {mean(nl_gi_times_cumul)} s, nl goa {mean(nl_goa_times_cumul)} s, nl lc2 {mean(nl_lc2_times_cumul)}"
    )


def run_group_experiments(group_path, cases, in_paths, out, skip_compute=False):
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
        for sigma in [0.0, 0.5, 1.0, 1.5, 2.0]:
            sigma_folder = f"sigma_{str(sigma)}"
            sigma_output_folder = os.path.join(out, folder, sigma_folder)
            os.makedirs(sigma_output_folder, exist_ok=True)
            for case in cases:
                if skip_compute:
                    continue
                case_path = os.path.join(group_path, case)
                case_output_path = os.path.join(sigma_output_folder, case)
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
                start = time.perf_counter()
                register(
                    fixed, moving, tag, case_output_path, sigma, force_rigid=force_rigid
                )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            plot_results(
                sigma_output_folder, f"{folder} transformation, sigma = {sigma}"
            )
    for folder, affine_folder, register, times in zip(
        ["nonlinear_gi", "nonlinear_goa", "nonlinear_lc2"],
        ["affine_gi", "affine_goa", "affine_lc2"],
        [nonlinear_gi, nonlinear_goa, nonlinear_lc2],
        [nonlinear_gi_times, nonlinear_goa_times, nonlinear_lc2_times],
    ):
        for sigma in [0.5, 1.0]:
            sigma_folder = f"sigma_{str(sigma)}"
            sigma_output_folder = os.path.join(out, folder, sigma_folder)
            os.makedirs(sigma_output_folder, exist_ok=True)
            for case in cases:
                if skip_compute:
                    continue
                case_path = os.path.join(group_path, case)
                case_output_path = os.path.join(sigma_output_folder, case)
                affine_matrix = os.path.join(
                    out, affine_folder, sigma_folder, case, "transformation.csv"
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
                start = time.perf_counter()
                register(
                    fixed,
                    moving,
                    tag,
                    affine_matrix,
                    case_output_path,
                    sigma,
                    bspline_spacing,
                    force_rigid=force_rigid,
                    dynamic_sampling=dynamic_sampling,
                )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            plot_results(
                sigma_output_folder, f"{folder} transformation, sigma = {sigma}", nonlinear=True
            )

    return affine_gi_times, affine_goa_times, affine_lc2_times, nonlinear_gi_times, nonlinear_goa_times, nonlinear_lc2_times


if __name__ == "__main__":
    data_path = os.path.abspath("/data")
    main(data_path)
