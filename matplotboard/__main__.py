from . import render, generate_report, Figure, __version__

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Matplotboard Command Line Utility")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument("--dir", help="Directory containing png files to put in a dump")

    args = parser.parse_args()

    if args.dir:
        from glob import glob
        from os.path import split, splitext

        figures = {}

        for f in glob("figs/*.png"):
            _, fname = split(f)
            fig_name, ext = splitext(fname)
            figures[fig_name] = Figure()
            figures[fig_name].orig_file = f

        render(figures)
        generate_report(figures, "Report")
