import matplotlib.pyplot as plt
from sklearn.metrics import auc
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

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

    h2_old_color = 'blue'
    h2_new_color = 'red'

    h2_old_x = list((0.6 + i/27 for i in range(11)))
    h2_old_y = list((0.9 - (i / 20) ** 2 for i in range(11)))
    h2_new_x = list((0.8 + i/50 for i in range(11)))
    h2_new_y = list((0.9 - (i / 22) ** 2 for i in range(11)))
    h1_acc = 0.6
    h1_y = [h1_acc, h1_acc]
    h1_x = [h2_old_x[0], h2_new_x[-1]]
    # h2_new_x = list((x/60+0.6 for x in range(20)))
    # h2_new_x = list((x/150+0.6 for x in range(20)))
    # h2_new_y = list((0.9 - (x / 60) ** 2 for x in range(20)))
    # ax.plot(h2_old_x, h2_old_y, 'b', marker='.', label='h2', markersize=8, linewidth=2)
    ax.fill_between(h2_old_x, h2_old_y, [h1_acc] * len(h2_old_x), facecolor=h2_old_color, alpha=0.1)
    ax.fill_between([h2_old_x[0]] + h2_new_x, [h2_new_y[0]] + h2_new_y, [h1_acc] * (len(h2_new_x) + 1),
                    facecolor=h2_new_color, alpha=0.1)
    ax.plot([h2_old_x[0], h2_new_x[0]], [h2_new_y[0], h2_new_y[0]], color=h2_new_color, dashes=[8, 2])
    ax.plot(h2_new_x, h2_new_y, h2_new_color, marker='o', markersize=6, label='h2 personalized')
    ax.plot(h2_old_x, h2_old_y, h2_old_color, marker='s', markersize=6, label='h2 baseline')
    ax.plot(h1_x, h1_y, 'k', dashes=[8, 2], label='h1 (pre-update)')
    # ax.plot(h1_x, h1_y, 'k', dashes=[10, 1, 2, 1], label='h1 (pre-update)')
    # ax.plot(h2_new_x, h2_new_y, 'r', label='h2')
    ax.set_xlabel('compatibility')
    ax.set_ylabel('performance')
    # ax.legend(('initial model', 'updated model', "new updated model"), loc='center left')
    # ax.set_title('Performance / Compatibility tradeoff for a specific user')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0.05))
    plt.savefig('sim plot.png', bbox_inches='tight')
    plt.show()

    # h1_area = (h1_x[1] - h1_x[0]) * h1_acc
    # old_area = auc(h2_old_x, h2_old_y) - h1_area
    # new_area = auc(h2_new_x, h2_new_y) - h1_area
    # diff = old_area/new_area - 1
    # print(diff * 100)


custom_plot()
