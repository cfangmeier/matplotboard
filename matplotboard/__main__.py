from .version import __version__


def main():
    from argparse import ArgumentParser
    from . import render, generate_report, loc_fig

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
        from os.path import join, split, splitext

        figures = {}

        for f in glob(join(args.dir, "*.png")):
            _, fname = split(f)
            fig_name, ext = splitext(fname)
            figures[fig_name] = loc_fig(f)

        render(figures)
        generate_report(figures, "Report")


if __name__ == "__main__":
    main()
