from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotboard as mpb


@mpb.decl_fig
def cool_fig(func, scale, color="b"):
    xs = np.linspace(-scale, scale, 100)
    f = {
        "sin": lambda xs: np.sin(xs),
        "tan": lambda xs: np.tan(xs),
        "exp": lambda xs: np.exp(xs),
    }[func]
    ys = f(xs)
    plt.plot(xs, ys, color=color)


report = """\
Authors: Will Hunting
Date: December 2, 1997

# Report On Functions

## Introduction

As we all know, there are many functions. An example is the sine function seen below.

fig::sin_b_1

## Other Functions

However, there are many other functions such as the tangent or exponential.

<div class="row">
<div class="col-md-6 row_fig">
fig::tan_r_1|The rugged tangent function
</div>
<div class="col-md-6 row_fig">
fig::exp_g_2|The majestic exponential function
</div>
</div>

The decision of which function is best is up to *you*!

## Local Figures

I happened to have a couple *really* fantastic figures on my computer that I
want to include as well. How do I include them? It's easy! Just add them to
the list of figures with the `loc_fig` function and they will be marked to be
copied to the output directory. Here are a couple examples:

<div class="row">
<div class="col-md-6 row_fig">
fig::image8
</div>
<div class="col-md-6 row_fig">
fig::image10
</div>
</div>

"""


def main():
    mpb.configure(multiprocess=True)
    figures = {}

    for color, function, scale in product(
        "rbgk", ["sin", "tan", "exp"], np.linspace(1, 5, 5)
    ):
        figures[f"{function}_{color}_{int(scale)}"] = cool_fig(
            function, scale, color=color
        )
    figures["image8"] = mpb.loc_fig("figures/image8.png")
    figures["image10"] = mpb.loc_fig("figures/image10.png")

    mpb.render(figures)
    mpb.generate_report(figures, "Report", body=report)


if __name__ == "__main__":
    main()
