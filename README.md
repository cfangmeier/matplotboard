![Matplotboard: For your Health!](https://cfangmeier.github.io/matplotboard/matplotboard.png)

[![Build Status](https://travis-ci.com/cfangmeier/matplotboard.svg?branch=master)](https://travis-ci.com/cfangmeier/matplotboard)

A utility to generate html dashboards using matplotlib. Matplotboard makes it easy to
wrap your plotting functions and dump the plots into a searchable webpage or a markdown report. This is best
demonstrated with an example.


``` python
import numpy as np
import matplotlib.pyplot as plt
import matplotboard as mpb

@mpb.decl_fig
def cool_fig():
    xs = np.linspace(-10, 10, 100)
    ys = xs**2
    plt.plot(xs, ys)

if __name__ == '__main__':
    figures = {
        'cool_fig': cool_fig(),
    }

    mpb.render(figures)
    mpb.generate_report(figures, 'Report')
```
You can view the results [here](https://cfangmeier.github.io/matplotboard/example_01/dashboard/report.html). Let's walk through this one part at a time.

First, we import `numpy` and `matplotlib` for some calculations, and plotting,
respectively. As well as `matplotboard` itself.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotboard as mpb
```


`matplotboard` relies upon `matplotlib` for the underlying rendering engine so
other plotting libraries are not supported. However, wrappers around
`matplotlib` such as `seaborn` should work.

Next, we declare the function that is actually going to do the plotting.

```python
@mpb.decl_fig
def cool_fig():
    xs = np.linspace(-10,10, 100)
    ys = xs**2
    plt.plot(xs, ys)
```

The `decl_fig` decorator modifies the function to work with `matplotboard`. A
plotting function decorated with `decl_fig` must fulfill the following
contract:

  - A clean `figure` has been initiated and the function will do any plotting
    on that figure.
  - It is free to subdivide the figure into as many axes as required, but
    shouldn't create additional `Figure` objects.
  - The function can optionally return Markdown text that will be rendered
    along with the plot.
  - The function shouldn't call `savefig`. This is handled by `matplotboard`
    automatically.

Finally, we declare the actual figures that we want to generate, and tell `matplotboard` to render the figures and assemble them into an interactive webpage.

```python
if __name__ == '__main__':
    figures = {
        'cool_fig': cool_fig(),
    }

    mpb.render(figures)
    mpb.generate_report(figures, 'Report')
```

Both `render`, and `generate_report` take a dictionary as their first argument.
The dictionary keys are strings that are interpreted as the individual figure
names, and the dictionary values are the plots we want to generate. Note that
the function is called before inserting it into the dictionary. Due to the 
modification of the original function by the decorator, this doesn't actually
call the function yet, but bundles the function and any arguments together
into a `Figure` object which it then returns for later processing by `matplotboard`.

By writing plotting functions with arguments, a single function can be reused
to make many different plots. For example, you may have a dataset that is
divided into several categories and you would like to plot some variable for
each category. You could do this by writing one plotting function and calling
it with different arguments to specify each of the categories.

Try running the example. If everything works, there should be a new folder in
the current directory called `dashboard`, and within it an html file called
`report.html`. Open it with your browser to see a dashboard containing a single
plot. Try clicking on it for a zoomed view!

A single plot is not very interesting. Where `matplotboard` starts to really become useful is when you have lots of plots to generate. Check out the following example.

```python
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotboard as mpb

@mpb.decl_fig
def cool_fig(func, scale, color='b'):
    xs = np.linspace(-scale, scale, 100)
    f = {
        'sin': lambda xs: np.sin(xs),
        'tan': lambda xs: np.tan(xs),
        'exp': lambda xs: np.exp(xs),
    }[func]
    ys = f(xs)
    plt.plot(xs, ys, color=color)

if __name__ == '__main__':
    mpb.configure(multiprocess=True)
    figures = {}

    for color, function, scale in product('rbgk', ['sin', 'tan', 'exp'], np.linspace(1, 20, 20)):
        figures[f'{function}_{color}_{scale}'] = cool_fig(function, scale, color=color)


    mpb.render(figures)
    mpb.generate_report(figures, 'Report')
```

What's changed? You can view the page [here](https://cfangmeier.github.io/matplotboard/example_02/dashboard/report.html)

First of all, the plotting function has been enhanced to take a few arguments
that modify it's behavior. You can now specify whether you would like to plot
`sin`, `tan`, or `exp` as well as effectively set the x length scale.

Second, we now are programatically making all combinations of plotting color,
function, and scale with the `product` function and declaring a plot for each
combination. This comes down to `4*3*20=240` different plots. To speed things
up a bit, this example also switches on `matplotboard`'s multiprocessing
support. Try running this example and open up the resulting web-page just as
before. Note the pagination feature limiting the number of figures displayed at
once. Also, try selecting a plot and moving through the figures on the page
with the arrow keys. Finally, try out the filter box in the top right. A few
interesting searches may be "sin\_", "\_r\_", or "tan\_g\_9" to search for all
`sin` plots, all red plots, and just the `tan_g_9` plot, respectively.

For one final example, let's look at the support for writing reports the incorporate generated figures.

```python

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

if __name__ == "__main__":
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
```

See result of this example [here](https://cfangmeier.github.io/matplotboard/example_03/dashboard/report.html).

The `generate_report` function supports an optional `body` argument which
signals `matplotboard` to render the markdown into a report, rather than making
a simple plot dump. A special syntax is used for embedding generated figures.

```
fig::figure_name|Optional Caption
```

`Bootstrap` is included by default so multiple figures side-by-side are
possible by use of a `row` div as shown in the example.

In addition to including generated figures via the `fig::` construct, static
figures (such as diagrams or photographs) can be included via the `locfig::`
(think local figure) construct, where instead of the figure name, you specify
the path to the file. Finally, pictures out on the internet can be specified
via `extfig::`.
