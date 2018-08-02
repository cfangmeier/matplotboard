import numpy as np
import matplotlib.pyplot as plt
import matplotboard as mpb

@mpb.decl_fig
def cool_fig():
    xs = np.linspace(-10,10, 100)
    ys = xs**2
    plt.plot(xs, ys)

if __name__ == '__main__':
    figures = {
        'cool_fig': cool_fig,
    }

    mpb.render(figures)
    mpb.generate_report(figures, 'Report')
