# coding: utf-8
if __name__ == "__main__":  # noqa
    import matplotlib

    matplotlib.use("Agg")


import seaborn as sns
import matplotlib.pyplot as plt

from genesis.objects import get_data


def main(ds, min_thickness=None, as_pairgrid=False, exclude_thin=False, sharex=False):
    N_objects_orig = int(ds.object_id.count())
    ds = ds.dropna("object_id")
    N_objects_nonan = int(ds.object_id.count())
    print(
        "{} objects out of {} remain after ones with nan for length, width"
        " or thickness have been remove".format(N_objects_nonan, N_objects_orig)
    )

    if min_thickness:
        hue_label = "thickness > {}m".format(min_thickness)
        ds[hue_label] = ds.thickness > min_thickness
        if exclude_thin:
            ds = ds.where(ds[hue_label], drop=True)
            print(ds.object_id.count())
    else:
        hue_label = None

    if as_pairgrid:
        g = sns.pairplot(
            ds.to_dataframe(), vars=["length", "width", "thickness"], hue=hue_label
        )
    else:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

        for n, v in enumerate(["length", "width", "thickness"]):
            ax = axes[n]
            if not exclude_thin:
                _, bins, _ = ds[v].plot.hist(ax=ax)
            else:
                bins = None
            if hue_label:
                ds_ = ds.where(ds[hue_label], drop=True)
                ds_[v].plot.hist(ax=ax, bins=bins)
            ax.set_xlim(0, None)
            ax.set_title("")
    sns.despine()

    if sharex:
        [ax.set_xlim(0, ds.length.max()) for ax in g.axes.flatten()]


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument("base_name")
    argparser.add_argument("--objects", default="*")
    argparser.add_argument("--frac", default=0.9)
    argparser.add_argument("--min-thickness", default=None, type=float)
    argparser.add_argument("--exclude-thin", default=False, action="store_true")
    argparser.add_argument("--sharex", default=False, action="store_true")
    argparser.add_argument("--as-pairgrid", default=False, action="store_true")

    args = argparser.parse_args()

    base_name = args.base_name
    frac = args.frac

    ds = get_data(base_name=base_name, mask_identifier=args.objects)
    main(ds=ds)

    if args.objects != "*":
        identifier = "{}.{}".format(base_name, args.objects)
    else:
        identifier = "{}.all".format(base_name)

    plt.suptitle(identifier + "\n\n", y=1.05)
    if not args.as_pairgrid:
        plt.tight_layout()

    fn = "{}.minkowski_scales.pdf".format(identifier)

    plt.savefig(fn, bbox_inches="tight")
    print("Saved plot to `{}`".format(fn))
