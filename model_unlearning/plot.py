import matplotlib.pyplot as plt
import numpy as np

for title, dataset in zip(["CIFAR10", "CIFAR100"], ['cf10', 'cf100']):

    def load_results(method):
        arr = []
        for i in range(1):
            iter = i + 1
            arr.append(
                np.load(
                    f"results/{dataset}_ref_vs_{method}{iter}/distance.npy"))
        return np.stack(arr)

    unrelated = load_results("un")
    approx = load_results("app")
    direct = load_results("dir")

    def plot_fig(ax: plt.Axes,
                 arr,
                 label,
                 color=(0, 1, 0),
                 ann="unrelated",
                 offset=-0.09):
        x = [i for i in range(2, 201, 2)]
        # arr = np.log(1 + arr * 100)
        m = arr.mean(0)
        s = arr.std(0)
        print(s.shape)
        # print(s)
        ax.fill_between(x, m - s, m + s, facecolor=color + (0.5, ))

        ax.annotate(r'(%s: %f)' % (ann, m[-1]), (130, (m[65] + offset)),
                    color=color,
                    fontsize=18)
        ax.set_xlabel("Epoch(t)", fontsize=23)
        ax.set_ylabel("Distance", fontsize=23)
        # ax.set_yscale('logit')
        ax.plot(x, m, label=label, color=color)

        for l in ax.get_xticklabels():
            l.set_fontsize(18)
        for l in ax.get_yticklabels():
            l.set_fontsize(18)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    def float_color(r, g, b):
        return (r / 255, g / 255, b / 255)

    # avg_unrelated = unrelated.mean(0)
    # unrelated = unrelated / avg_unrelated
    # direct = direct / avg_unrelated
    # approx = approx / avg_unrelated
    plot_fig(ax,
             unrelated,
             color=float_color(0, 205, 205),
             label=r'$C^{(t)}_{ref} v.s. C_{unrelated}$',
             ann="unrelated",
             offset=0.08)
    plot_fig(ax,
             direct,
             color=float_color(106, 90, 205),
             label=r'$C^{(200)}_{ref} v.s. C^{(t)}_{direct}$',
             ann="direct",
             offset=-0.1)

    plot_fig(ax,
             approx,
             color=float_color(102, 205, 170),
             label=r'$C^{(200)}_{ref} v.s. C^{(t)}_{approx}$',
             ann="approx",
             offset=0.065)
    ax.set_title(title, fontsize=26)
    ax.legend(loc=3, fontsize=18)
    fig.tight_layout()
    fig.savefig(f'{dataset}.pdf')
