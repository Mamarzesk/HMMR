import os

from nonlinear_GI import nonlinear_gi
from nonlinear_GOA import nonlinear_goa
from nonlinear_LC2 import nonlinear_lc2
from plot_results import plot_results


def main(data_path, skip_compute=False):
    output_root = os.path.join(data_path, "grid_search")
    rigid_root = os.path.join(data_path, "all_methods_paper")
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
    rigid_list = [
        bite_g2_rigid,
        bite_g4_rigid,
        resect_pre_flair_rigid,
        resect_post_flair_rigid,
        resect_pre_t1_rigid,
        resect_post_t1_rigid,
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

    for group_path, case_list, in_paths, out, rigid in zip(
        group_paths, cases_list, in_paths_list, out_list, rigid_list
    ):
        run_group_experiments(
            group_path,
            case_list,
            in_paths,
            out,
            rigid=rigid,
            skip_compute=skip_compute,
        )


def run_group_experiments(
    group_path, cases, in_paths, out, rigid="rigid_paper", skip_compute=False
):
    sample_count = 10_000
    sigma = 0.5
    dynamic_sampling = True
    force_rigid = True
    bspline_spacing = [5.0, 10.0, 15.0]
    regularizers_test_vals = [50, 100, 150, 200]
    already_done_combos = [
        (5.0, 100),
        (10.0, 100),
    ]

    for folder, affine_folder, register in zip(
        ["nonlinear_gi", "nonlinear_goa", "nonlinear_lc2"],
        ["affine_gi", "affine_goa", "affine_lc2"],
        [nonlinear_gi, nonlinear_goa, nonlinear_lc2],
    ):
        for spacing in bspline_spacing:
            for regularizer in regularizers_test_vals:
                if (spacing, regularizer) in already_done_combos:
                    continue

                regularizer_folder = f"regularizer_{regularizer}"
                spacing_folder = f"spacing_{spacing}"
                grid_output_folder = os.path.join(
                    out, folder, spacing_folder, regularizer_folder
                )
                os.makedirs(grid_output_folder, exist_ok=True)
                for case in cases:
                    if skip_compute:
                        continue
                    case_path = os.path.join(group_path, case)
                    case_output_path = os.path.join(grid_output_folder, case)
                    affine_matrix = os.path.join(
                        rigid,
                        affine_folder,
                        f"sigma_{sigma:.1f}",
                        case,
                        "transformation.csv",
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
                    register(
                        fixed,
                        moving,
                        tag,
                        affine_matrix,
                        case_output_path,
                        sigma=1,
                        bspline_spacing=spacing,
                        force_rigid=force_rigid,
                        dynamic_sampling=dynamic_sampling,
                        samples_count=sample_count,
                        regularizer=regularizer,
                    )
                plot_results(
                    grid_output_folder,
                    f"{folder} transformation, regularizer = {regularizer}, bspline spacing = {spacing}",
                    nonlinear=True,
                )


if __name__ == "__main__":
    data_path = os.path.abspath("/data")
    main(data_path)
