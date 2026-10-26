"""Extract frozen BrainIAC features from the OASIS-1 FreeSurfer brain masks of the 226 study subjects.

The script reproduces the BrainIAC inference pipeline (github.com/AIM-KannLab/BrainIAC, non-commercial
licence) on skull-stripped FreeSurfer volumes:
  1. brainmask.mgz -> NIfTI;
  2. N4 bias-field correction (SimpleITK, as in BrainIAC's mri_preprocess_3d_simple.py);
  3. rigid (Euler3D) registration to BrainIAC's template temp_head.nii.gz with Mattes mutual information
     and the same optimiser settings as the BrainIAC script, resampled onto the template grid (1 mm);
  4. HD-BET is skipped, because brainmask.mgz is already skull-stripped;
  5. the released BrainIAC checkpoint (ViT-B backbone of the public repository) with BrainIAC's validation
     transform (whole volume resized to 96^3, z-scored over non-zero voxels) gives the 768-dimensional
     CLS-token feature of every subject.

Nothing is trained. The output CSV (subject_id + 768 features) is derived from OASIS data and is not
redistributed; scripts/probe.py turns it into out-of-fold predictions of a linear probe.

Usage (Colab or local):
  python brainiac_features.py --oasis-dir <dir with */mri/brainmask.mgz> --folds data/folds.csv \
      --brainiac-src <BrainIAC repo>/src --checkpoint <BrainIAC.ckpt> --out-dir <output dir> [--no-n4] [--limit N]
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

N4_SHRINK = 4  # fit the bias field on a 4x downsampled image and apply it at full resolution


def find_brainmasks(oasis_dir, subject_ids):
    """Map subject_id -> path of mri/brainmask.mgz, walking the OASIS FreeSurfer tree once."""
    wanted, found = set(subject_ids), {}
    for root, _dirs, files in os.walk(oasis_dir):
        if "brainmask.mgz" in files and os.path.basename(root) == "mri":
            sid = os.path.basename(os.path.dirname(root))
            if sid in wanted:
                found[sid] = os.path.join(root, "brainmask.mgz")
    return found


def mgz_to_sitk(path):
    img = nib.load(path)
    nii = nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32), img.affine)
    tmp = Path(path).with_suffix(".tmp.nii.gz")
    nib.save(nii, tmp)
    out = sitk.ReadImage(str(tmp), sitk.sitkFloat32)
    tmp.unlink()
    return out


def n4_correct(image):
    """N4 as in the BrainIAC script (default parameters, Otsu mask), but fitted on a shrunken copy for speed."""
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    small = sitk.Shrink(image, [N4_SHRINK] * 3)
    small_mask = sitk.Shrink(mask, [N4_SHRINK] * 3)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(small, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias)
    return sitk.Cast(corrected, sitk.sitkFloat32)


def load_template(path):
    """BrainIAC's temp_head.nii.gz, resampled to 1 mm isotropic exactly as its preprocessing script does."""
    fixed = sitk.ReadImage(path, sitk.sitkFloat32)
    size, spacing = fixed.GetSize(), fixed.GetSpacing()
    new_size = [int(round(size[i] * spacing[i])) for i in range(3)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing((1.0, 1.0, 1.0))
    resample.SetSize(new_size)
    resample.SetOutputOrigin(fixed.GetOrigin())
    resample.SetOutputDirection(fixed.GetDirection())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputPixelType(sitk.sitkFloat32)
    return resample.Execute(fixed)


def register_rigid(fixed, moving):
    """Euler3D registration with the BrainIAC settings; returns the resampled image and the final metric."""
    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.01, seed=0)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                         convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial)
    final = method.Execute(fixed, moving)
    resampled = sitk.Resample(moving, fixed, final, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return resampled, float(method.GetMetricValue())


def preprocess_subject(mgz_path, template, out_path, use_n4=True):
    moving = mgz_to_sitk(mgz_path)
    if use_n4:
        moving = n4_correct(moving)
    registered, metric = register_rigid(template, moving)
    sitk.WriteImage(registered, str(out_path))
    return metric


def qc_montage(nifti_paths, out_png, n=12):
    """Mid-axial and mid-coronal slices of up to n registered volumes, for a visual registration check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = list(nifti_paths)[:n]
    fig, axes = plt.subplots(2, len(paths), figsize=(1.6 * len(paths), 3.4))
    axes = np.atleast_2d(axes)
    for j, p in enumerate(paths):
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))  # z, y, x
        axes[0, j].imshow(vol[vol.shape[0] // 2], cmap="gray")
        axes[1, j].imshow(vol[:, vol.shape[1] // 2, :], cmap="gray", origin="lower")
        axes[0, j].set_title(Path(p).name.split(".")[0][:12], fontsize=6)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def extract_features(nifti_paths, brainiac_src, checkpoint, device):
    """Frozen BrainIAC ViT-B features with BrainIAC's validation transform.

    Returns two 768-dimensional feature sets per volume: BrainIAC's own representation, the first token
    of the final layer (what ViTBackboneNet.forward returns and what its downstream heads were trained on),
    and the mean over all 216 patch tokens as a sensitivity variant."""
    import torch
    sys.path.insert(0, str(brainiac_src))
    from dataset import get_validation_transform  # noqa: E402  (BrainIAC repository)

    backbone = load_backbone(checkpoint).to(device).eval()
    transform = get_validation_transform()
    token0, pooled = [], []
    with torch.no_grad():
        for p in nifti_paths:
            x = torch.as_tensor(transform({"image": str(p)})["image"]).unsqueeze(0).to(device)
            tokens = backbone(x)[0]  # (1, 216, 768): no class token, as in the released model
            token0.append(tokens[:, 0].squeeze(0).cpu().numpy())
            pooled.append(tokens.mean(dim=1).squeeze(0).cpu().numpy())
    return np.vstack(token0), np.vstack(pooled)


def load_backbone(checkpoint):
    """The BrainIAC ViT-B backbone (same constructor arguments as BrainIAC's ViTBackboneNet) with the
    SimCLR checkpoint weights. MONAI versions after 1.3 add `norm_cross_attn` layers to every transformer
    block; they are unused without cross-attention, so only those keys may be missing from the checkpoint."""
    import torch
    from monai.networks.nets import ViT
    backbone = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16), hidden_size=768,
                   mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    weights = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    result = backbone.load_state_dict(weights, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if "norm_cross_attn" not in k]
    if unexpected or missing:
        raise SystemExit(f"checkpoint does not match the BrainIAC ViT-B: unexpected {unexpected[:5]}, missing {missing[:5]}")
    print(f"BrainIAC backbone loaded: {len(weights)} tensors"
          + (f" ({len(result.missing_keys)} unused norm_cross_attn tensors left at initialisation)" if result.missing_keys else ""))
    return backbone


def write_features(path, subjects, feats):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id"] + [f"f{j}" for j in range(feats.shape[1])])
        for sid, row in zip(subjects, feats):
            w.writerow([sid] + [f"{v:.6g}" for v in row])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oasis-dir", type=Path, required=True, help="root of the OASIS-1 FreeSurfer tree")
    ap.add_argument("--folds", type=Path, required=True, help="data/folds.csv with the 226 subject IDs")
    ap.add_argument("--brainiac-src", type=Path, required=True, help="the src/ folder of the BrainIAC repository")
    ap.add_argument("--checkpoint", type=Path, required=True, help="BrainIAC.ckpt")
    ap.add_argument("--template", type=Path, default=None,
                    help="registration template (default: <brainiac-src>/preprocessing/atlases/temp_head.nii.gz)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-n4", action="store_true", help="skip N4 bias-field correction")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N subjects (for tests)")
    ap.add_argument("--skip-existing", action="store_true", help="reuse registered volumes already in out-dir")
    ap.add_argument("--stage", choices=["all", "preprocess", "features"], default="all",
                    help="run only the CPU preprocessing or only the GPU feature extraction")
    args = ap.parse_args()

    template_path = args.template or args.brainiac_src / "preprocessing" / "atlases" / "temp_head.nii.gz"
    out_dir = args.out_dir
    (out_dir / "registered").mkdir(parents=True, exist_ok=True)

    with open(args.folds, newline="") as f:
        subjects = [row["subject_id"] for row in csv.DictReader(f)]
    if args.limit:
        subjects = subjects[:args.limit]
    masks = find_brainmasks(args.oasis_dir, subjects)
    missing = [s for s in subjects if s not in masks]
    print(f"{len(masks)}/{len(subjects)} brainmask.mgz found under {args.oasis_dir}")
    if missing:
        print(f"missing ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    subjects = [s for s in subjects if s in masks]

    registered = [out_dir / "registered" / f"{sid}.nii.gz" for sid in subjects]
    if args.stage in ("all", "preprocess"):
        template = load_template(str(template_path))
        print(f"template {template_path.name}: size {template.GetSize()}, spacing {template.GetSpacing()}")
        metrics, t0 = {}, time.time()
        for i, (sid, out_path) in enumerate(zip(subjects, registered), 1):
            if args.skip_existing and out_path.exists():
                continue
            metrics[sid] = preprocess_subject(masks[sid], template, out_path, use_n4=not args.no_n4)
            if i % 10 == 0 or i == len(subjects):
                print(f"  registered {i}/{len(subjects)} ({(time.time() - t0) / i:.1f} s per subject)")
        if metrics:
            with open(out_dir / "registration_metric.csv", "a", newline="") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(["subject_id", "mattes_mi"])
                w.writerows(sorted(metrics.items()))
            values = np.array(list(metrics.values()))
            print(f"Mattes MI (more negative = better): median {np.median(values):.3f}, "
                  f"worst {values.max():.3f} ({max(metrics, key=metrics.get)})")
        qc_montage(registered, out_dir / "qc_registration.png")

    if args.stage in ("all", "features"):
        missing = [p.name for p in registered if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} registered volumes missing, e.g. {missing[:3]}; run --stage preprocess first")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"extracting features on {device} ...")
        token0, pooled = extract_features(registered, args.brainiac_src, args.checkpoint, device)
        write_features(out_dir / "brainiac_features.csv", subjects, token0)
        write_features(out_dir / "brainiac_features_meanpool.csv", subjects, pooled)
        print(f"features {token0.shape} written to {out_dir}: brainiac_features.csv (BrainIAC token) and "
              f"brainiac_features_meanpool.csv (mean of 216 tokens); token-feature norm median "
              f"{np.median(np.linalg.norm(token0, axis=1)):.2f}")


if __name__ == "__main__":
    main()
