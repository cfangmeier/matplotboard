# Matplotboard

A utility to generate html dashboards using matplotlib. Matplotboard makes it easy to
wrap your plotting functions and embed them into a Markdown document. This is best
demonstrated with an example.


``` python
import numpy as np
import matplotlib.pyplot as plt

from matplotboard import (decl_fig, render, generate_report)


@decl_fig
def cool_fig():
    xs = np.linspace(-10,10, 100)
    ys = xs**2
    plt.plot(xs, ys)


@decl_fig
def fig_with_args(amp, freq):
    '''
    A plot of a sine wave with configurable amplitude and frequency.
    '''
    xs = np.linspace(-np.pi, np.pi, 100)
    ys = amp*np.sin(xs*freq)
    plt.plot(xs, ys)


if __name__ == '__main__':
    figures = {'my_cool_fig': cool_fig,
               'slow': (fig_with_args, (1, np.pi)),
               'fast': (fig_with_args, (0.75, 3*np.pi)),
               }

    render(figures)
    generate_report(figures, 'Report',
                    source=__file__,
                    body="""
# Making **Awesome Dashboards**

Sometimes you just want to push out a static html page with plots and relevant
commentary. For example, what does a parabola look like?

fig::my_cool_fig

There we go, `matplotboard` also supports plotting functions that take arguments.

fig::slow
fig::fast

    """)
```
You can view the fantastic generated report [here](https://cfangmeier.github.io/matplotboard/report.html)
