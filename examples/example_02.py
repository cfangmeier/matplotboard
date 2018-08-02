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
        figures[f'{function}_{color}_{int(scale)}'] = cool_fig, (function, scale), dict(color=color)


    mpb.render(figures)
    mpb.generate_report(figures, 'Report')
