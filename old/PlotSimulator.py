import matplotlib.pyplot as plt
from sklearn.metrics import auc


def custom_plot():

    fig, ax = plt.subplots(1)
    # fig.patch.set_facecolor('green')

    # i = 0
    # fig.axhspan(i, i + .2, facecolor='0.2', alpha=0.5)
    # fig.axvspan(i, i + .5, facecolor='b', alpha=0.5)

    # for ax in axs.ravel():
    # ax.axis('off')

    # ax.edgecolor('green')

    # plt.setp(ax.get_xticklabels(), visible=False)
    # plt.setp(ax.get_yticklabels(), visible=False)
    # ax.tick_params(axis='both', which='both', length=0)

    h1_acc = 0.6
    h1_x = [0.6, 1.0]
    h1_y = [h1_acc, h1_acc]
    ax.plot(h1_x, h1_y, 'k--', label='h1')
    h2_old_x = list((x/50+0.6 for x in range(20)))
    h2_new_x = list((x/60+0.6 for x in range(20)))
    # h2_new_x = list((x/150+0.6 for x in range(20)))
    h2_old_y = list((0.9 - (x / 40) ** 2 for x in range(20)))
    h2_new_y = list((0.9 - (x / 60) ** 2 for x in range(20)))
    # ax.plot(h2_old_x, h2_old_y, 'b', marker='.', label='h2', markersize=8, linewidth=2)
    ax.fill_between(h2_old_x, h2_old_y, [h1_acc] * len(h2_old_x), facecolor='b', alpha=0.15)
    ax.fill_between(h2_new_x, h2_new_y, [h1_acc] * len(h2_new_x), facecolor='r', alpha=0.15)
    ax.plot(h2_old_x, h2_old_y, 'b', label='h2', markersize=8, linewidth=2)
    ax.plot(h2_new_x, h2_new_y, 'r', label='h2')
    ax.set_xlabel('compatibility')
    ax.set_ylabel('accuracy')
    # ax.legend(('initial model', 'updated model', "new updated model"), loc='center left')
    ax.set_title('Performance / Compatibility tradeoff for a specific user')
    plt.savefig('sim plot.png', bbox_inches='tight')
    plt.show()

    h1_area = (h1_x[1] - h1_x[0]) * h1_acc
    old_area = auc(h2_old_x, h2_old_y) - h1_area
    new_area = auc(h2_new_x, h2_new_y) - h1_area
    diff = old_area/new_area - 1
    print(diff * 100)


custom_plot()
