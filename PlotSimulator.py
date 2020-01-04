import matplotlib.pyplot as plt


def custom_plot():

    fig, ax = plt.subplots(1)
    # fig.patch.set_facecolor('green')

    # i = 0
    # fig.axhspan(i, i + .2, facecolor='0.2', alpha=0.5)
    # fig.axvspan(i, i + .5, facecolor='b', alpha=0.5)

    # for ax in axs.ravel():
    # ax.axis('off')

    # ax.edgecolor('green')

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

    h1_x = [0.6, 1.0]
    h1_y = [0.6, 0.6]
    ax.plot(h1_x, h1_y, 'k--', label='h1')
    h2_x = list((x/50+0.6 for x in range(20)))
    h2_old_y = list((0.9 - (x / 40) ** 2 for x in range(20)))
    h2_new_y = list((0.9 - (x / 60) ** 2 for x in range(20)))
    ax.plot(h2_x, h2_old_y, 'b', marker='.', label='h2', markersize=8, linewidth=2)
    ax.plot(h2_x, h2_new_y, 'r', marker='.', label='h2')
    # ax.set_xlabel('compatibility')
    # ax.set_ylabel('accuracy')
    # ax.legend(('initial model', 'updated model', "new updated model"), loc='center left')
    # ax.set_title('Performance / Compatibility tradeoff for a specific user')

    plt.show()


custom_plot()
