import csv
import numpy as np
import os
from sklearn.metrics import auc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ast


def binarize(path, bin_count, diss_types):
    with open(path + ".csv", 'r', newline='') as file_in:
        with open(path + "_" + str(bin_count) + "_bins.csv", 'w', newline='') as file_out:
            reader = csv.reader(file_in)
            writer = csv.writer(file_out)

            # save rows, get min and max com
            rows = []
            in_header = True
            in_first = True
            min_com = 0
            max_com = 0
            base_acc = 0
            for row in reader:
                if in_header:
                    in_header = False
                    writer.writerow(row)
                    continue
                rows += [row]
                com = float(row[0])
                if in_first:
                    in_first = False
                    min_com = com
                    max_com = com
                    base_acc = float(row[1])
                    continue
                if com < min_com:
                    min_com = com
                if com > max_com:
                    max_com = com

            # init bins
            bin_width = (max_com - min_com) / bin_count
            bins = []
            for i in range(bin_count):
                bin = []
                for j in range(diss_types):
                    bin += [[]]
                bins += [bin]

            # fill bins
            for row in rows:
                com = float(row[0])
                i = int((com - min_com) / bin_width)
                if i == bin_count:
                    i = bin_count - 1
                for j in range(diss_types):
                    if row[2 + j] != "":
                        bins[i][j] += [(float(row[2 + j]))]

            # write file
            for i in range(len(bins)):
                bin = bins[i]
                row = [str(min_com + (i + 0.5) * bin_width), str(base_acc)]
                bin_empty = True
                for j in range(diss_types):
                    if len(bin[j]) != 0:
                        bin_empty = False
                        row += [str(np.mean(bin[j]))]
                    else:
                        row += [""]
                if not bin_empty:
                    writer.writerow(row)


def column_splitter(path, column, values_to_take=1):
    with open(path + ".csv", 'r', newline='') as file_in:
        with open(path + "_" + column + "_splitted.csv", 'w', newline='') as file_out:
            reader = csv.reader(file_in)
            writer = csv.writer(file_out)
            first = True
            column_idx = 0
            skill_name_column = 0
            user_id_column = 0
            skills_dict = {}
            for row in reader:
                if first:
                    first = False
                    writer.writerow(['user_id', 'original_' + column, column])
                    for i in range(len(row)):
                        name = row[i]
                        if name == column:
                            column_idx = i
                        elif name == 'user_id':
                            user_id_column = i
                        # elif name == 'skill_name':
                        #     skill_name_column = i
                else:
                    full_value = row[column_idx]
                    values = full_value.split(',')
                    # skill_names_string = row[skill_name_column]
                    # skill_names = skill_names_string.split(',')
                    user_id = row[user_id_column]
                    for i in range(len(values)):
                        if i == values_to_take:
                            break
                        value = values[i]
                        # skill_name = skill_names[i]
                        # if skill_id not in skills_dict:
                        #     skills_dict[skill_id] = skill_name
                        writer.writerow([user_id, full_value, value])

    # with open(path + "_skill_dictionary.csv", 'w', newline='') as file_out:
    #     writer = csv.writer(file_out)
    #     writer.writerow(['skill_id', 'skill_name'])
    #     for skill_id, skill_name in skills_dict.items():
    #         writer.writerow([skill_id, skill_name])


def csv_line_to_comma_delimited():
    path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\episode_run_0005_part_00'
    with open(path + ".csv", 'r', newline='') as file_in:
        with open(path + "_fixed.csv", 'w', newline='') as file_out:
            writer = csv.writer(file_out)
            cols = ['account_id',
                    'episode_id',
                    'session_key',
                    'discriminator',
                    'user_type',
                    'class_id',
                    'teacher_id',
                    'grade_code',
                    'locale_id',
                    'episode_slug',
                    'envelop_version',
                    'envelope',
                    'start_time',
                    'finish_time',
                    'last_submitted_index',
                    'time_spent_sec',
                    'activity_context',
                    'score',
                    'time_finished_sec',
                    'is_finished',
                    'problem_id',
                    'correct_answers_percentage',
                    'last_discriminator',
                    'last_in_session',
                    'before_replay',
                    'etl_time',
                    'client_ip']
            cols_used = ['account_id',
                         'is_finished',
                         'user_type',
                         'grade_code',
                         'activity_context',
                         'correct_answers_percentage',
                         'time_spent_sec']
            writer.writerow(cols_used)
            cols_used_idxs = []
            for i in range(len(cols)):
                col = cols[i]
                if col in cols_used:
                    cols_used_idxs += [i]

            # i = 0
            for row in file_in:
                # i += 1
                # if i > 10:
                #     break
                raw_values = row.split('|')
                values = list((x[1:-1] for x in raw_values))
                writer.writerow([values[i] for i in cols_used_idxs])


def csv_splitter(path, column_idxs, max_rows_per_part=500000):
    with open(path + ".csv", 'r', newline='', encoding='utf-8') as file_in:
        reader = csv.reader(file_in)
        first = True
        header = []
        part = 0
        done = False
        while not done:
            part += 1
            with open(path + "_part_" + str(part) + ".csv", 'w', newline='', encoding='utf-8') as file_out:
                part_end_reached = False
                writer = csv.writer(file_out)
                first_in_part = True
                rows_in_part = 0
                while not part_end_reached:
                    try:
                        raw_row = next(reader)
                        row = [raw_row[i] for i in column_idxs]
                        rows_in_part += 1
                        if first:
                            first = False
                            header = row
                        elif first_in_part:
                            first_in_part = False
                            writer.writerow(header)
                            writer.writerow(row)
                        else:
                            writer.writerow(row)
                        if rows_in_part == max_rows_per_part:
                            part_end_reached = True
                    except StopIteration:
                        done = True
                        break


def from_file_names_to_csv(folder_path, file_path):
    with open(file_path, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(['user_id', 'cos_sim'])
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                strings = name.split('_')
                user_id = strings[4].split('.')[0]
                cos_sim = strings[1]
                writer.writerow([user_id, cos_sim])


def make_monotonic(x, y):
    i = 0
    while i < len(x) - 1:
        if x[i + 1] < x[i]:
            del x[i + 1]
            del y[i + 1]
        else:
            i += 1


def cutoff(x, y, cutoff_x):
    i = len(x)-1
    while i >= 0:
        if x[i] > cutoff_x:
            del x[i]
            del y[i]
            i -= 1
        else:
            break


def area_calculator(log_path, hybrid_stat_log_path, hybrid_nn_log_path, plots_dir, L1_log_path=None,
                    cut_by_min=False, skip_L_models=False, only_L1=False):

    auc_path = plots_dir + '\\auc_from_log.csv'
    log_df = pd.read_csv(log_path)
    hybrid_stat_log_df = pd.read_csv(hybrid_stat_log_path)
    hybrid_nn_log_df = pd.read_csv(hybrid_nn_log_path)
    if only_L1:
        L1_log_df = pd.read_csv(L1_log_path)

    log_by_users = log_df.groupby(['user_id'])
    hybrid_stat_log_by_users = hybrid_stat_log_df.groupby(['user_id'])
    hybrid_nn_log_by_users = hybrid_nn_log_df.groupby(['user_id'])
    if only_L1:
        L1_log_by_users = L1_log_df.groupby(['user_id'])

    user_ids = log_df.user_id.unique()

    with open(auc_path, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        # writer.writerow(['train frac', 'user_id', 'instances', 'cos sim', 'train seed', 'comp range', 'acc range',
        #                  'h1 acc', 'no hist area', 'L0 area', 'L1 area', 'L2 area', 'hybrid_stat area', 'hybrid_nn area'])
        writer.writerow(['train frac', 'user_id', 'instances', 'train seed', 'comp range', 'acc range',
                         'h1 acc', 'no hist area', 'L0 area', 'L1 area', 'L2 area', 'hybrid_stat area',
                         'hybrid_nn area'])

        user_count = 0
        for user_id in user_ids:
            user_count += 1
            print(str(user_count)+'/'+str(len(user_ids)) + ' user ' + str(user_id))

            user_log = log_by_users.get_group(user_id).reset_index(drop=True)
            user_hybrid_stat_log = hybrid_stat_log_by_users.get_group(user_id).reset_index(drop=True)
            user_hybrid_nn_log = hybrid_nn_log_by_users.get_group(user_id).reset_index(drop=True)
            if only_L1:
                user_L1_log = L1_log_by_users.get_group(user_id).reset_index(drop=True)

            x = user_log.loc[0]

            history_len = x['instances']
            seed = x['train seed']
            com_range = x['comp range']
            auc_range = x['acc range']
            h1_acc = x['h1 acc']

            # sim = '%.4f' % (user_hybrid_stat_log.loc[0]['sim'],)

            row = [x['train frac'],
                   str(user_id),
                   history_len,
                   # sim,
                   seed,
                   com_range,
                   auc_range,
                   h1_acc]

            no_hist_x = user_log['no hist x'].tolist()
            no_hist_y = user_log['no hist y'].tolist()
            L0_x = user_log['L0 x'].tolist()
            L0_y = user_log['L0 y'].tolist()
            if not only_L1:
                L1_x = user_log['L1 x'].tolist()
                L1_y = user_log['L1 y'].tolist()
            else:
                L1_x = user_L1_log['L1 x'].tolist()
                L1_y = user_L1_log['L1 y'].tolist()
            L2_x = user_log['L2 x'].tolist()
            L2_y = user_log['L2 y'].tolist()
            hybrid_stat_x = user_hybrid_stat_log['hybrid x'].tolist()
            hybrid_stat_y = user_hybrid_stat_log['hybrid y'].tolist()
            hybrid_nn_x = user_hybrid_nn_log['hybrid x'].tolist()
            hybrid_nn_y = user_hybrid_nn_log['hybrid y'].tolist()

            min_x = min(user_log[['no hist x', 'L0 x', 'L1 x', 'L2 x']].values.min(),
                        user_hybrid_stat_log['hybrid x'].values.min(),
                        user_hybrid_nn_log['hybrid x'].values.min())

            mono_no_hist_x = [min_x] + no_hist_x.copy()
            mono_no_hist_y = [no_hist_y[0]] + no_hist_y.copy()
            mono_L0_x = [min_x] + L0_x.copy()
            mono_L0_y = [L0_y[0]] + L0_y.copy()
            mono_L1_x = [min_x] + L1_x.copy()
            mono_L1_y = [L1_y[0]] + L1_y.copy()
            mono_L2_x = [min_x] + L2_x.copy()
            mono_L2_y = [L2_y[0]] + L2_y.copy()
            mono_hybrid_stat_x = [min_x] + hybrid_stat_x.copy()
            mono_hybrid_stat_y = [hybrid_stat_y[0]] + hybrid_stat_y.copy()
            mono_hybrid_nn_x = [min_x] + hybrid_nn_x.copy()
            mono_hybrid_nn_y = [hybrid_nn_y[0]] + hybrid_nn_y.copy()

            mono_xs = [mono_no_hist_x, mono_L0_x, mono_L1_x, mono_L2_x, mono_hybrid_stat_x, mono_hybrid_nn_x]
            mono_ys = [mono_no_hist_y, mono_L0_y, mono_L1_y, mono_L2_y, mono_hybrid_stat_y, mono_hybrid_nn_y]

            for i in range(len(mono_xs)):
                make_monotonic(mono_xs[i], mono_ys[i])

            if cut_by_min:
                cutoff_x = 1
                for mono_x in mono_xs:
                    max_mono_x = max(mono_x)
                    cutoff_x = min(cutoff_x, max_mono_x)

                for i in range(len(mono_xs)):
                    cutoff(mono_xs[i], mono_ys[i], cutoff_x)

            mono_no_hist_x += [mono_no_hist_x[-1], 1]
            mono_no_hist_y += [h1_acc, h1_acc]
            mono_L0_x += [mono_L0_x[-1], 1]
            mono_L0_y += [h1_acc, h1_acc]
            mono_L1_x += [mono_L1_x[-1], 1]
            mono_L1_y += [h1_acc, h1_acc]
            mono_L2_x += [mono_L2_x[-1], 1]
            mono_L2_y += [h1_acc, h1_acc]
            mono_hybrid_stat_x += [mono_hybrid_stat_x[-1], 1]
            mono_hybrid_stat_y += [h1_acc, h1_acc]
            mono_hybrid_nn_x += [mono_hybrid_nn_x[-1], 1]
            mono_hybrid_nn_y += [h1_acc, h1_acc]

            h1_area = (1 - min_x) * h1_acc
            no_hist_area = auc(mono_no_hist_x, mono_no_hist_y) - h1_area
            L0_area = auc(mono_L0_x, mono_L0_y) - h1_area
            L1_area = auc(mono_L1_x, mono_L1_y) - h1_area
            L2_area = auc(mono_L2_x, mono_L2_y) - h1_area
            hybrid_stat_area = auc(mono_hybrid_stat_x, mono_hybrid_stat_y) - h1_area
            hybrid_nn_area = auc(mono_hybrid_nn_x, mono_hybrid_nn_y) - h1_area

            row += [no_hist_area, L0_area, L1_area, L2_area, hybrid_stat_area, hybrid_nn_area]
            writer.writerow(row)

            h1_x = [min_x, 1]
            h1_y = [h1_acc, h1_acc]
            plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')
            plt.text(min_x, h1_acc * 1.005, 'h1')

            if not skip_L_models:
                if not only_L1:
                    plt.plot(no_hist_x, no_hist_y, 'b', marker='.', linewidth=3, markersize=22,
                             label='h2 not using history')
                    plt.plot(hybrid_stat_x, hybrid_stat_y, 'g', marker='.', linewidth=3, markersize=18,
                             label='h2 hybrid stat')
                    plt.plot(hybrid_nn_x, hybrid_nn_y, 'seagreen', marker='.', linewidth=3, markersize=18,
                             label='h2 hybrid nn')
                    plt.plot(L0_x, L0_y, 'r', marker='.', linewidth=3, markersize=14,
                             label='h2 using L0')
                    plt.plot(L1_x, L1_y, 'm', marker='.', linewidth=3, markersize=10,
                             label='h2 using L1')
                    plt.plot(L2_x, L2_y, 'orange', marker='.', linewidth=3, markersize=6,
                             label='h2 using L2')
                else:
                    plt.plot(no_hist_x, no_hist_y, 'b', marker='.', linewidth=3, markersize=18,
                             label='h2 not using history')
                    plt.plot(hybrid_stat_x, hybrid_stat_y, 'g', marker='.', linewidth=3, markersize=16,
                             label='h2 hybrid stat')
                    plt.plot(hybrid_nn_x, hybrid_nn_y, 'seagreen', marker='.', linewidth=3, markersize=14,
                             label='h2 hybrid nn')
                    plt.plot(L1_x, L1_y, 'm', marker='.', linewidth=3, markersize=12,
                             label='h2 using L1')
            else:
                plt.plot(no_hist_x, no_hist_y, 'b', marker='.', linewidth=3, markersize=18,
                         label='h2 not using history')
                plt.plot(hybrid_stat_x, hybrid_stat_y, 'g', marker='.', linewidth=3, markersize=16,
                         label='h2 hybrid stat')
                plt.plot(hybrid_nn_x, hybrid_nn_y, 'seagreen', marker='.', linewidth=3, markersize=14,
                         label='h2 hybrid nn')

            plt.xlabel('compatibility')
            plt.ylabel('accuracy')
            # plt.legend()

            # plt.title(
            #     'user=' + str(user_id) + ' hist_len=' + str(history_len) + ' split=' + str(x['train frac'])
            #     + ' sim=' + sim + '\n\n')

            plt.title(
                'user=' + str(user_id) + ' hist_len=' + str(int(history_len)) + ' split=' + str(x['train frac']) + '\n\n')

            areas = [no_hist_area, hybrid_stat_area, hybrid_nn_area]
            col_labels = ['no hist', 'hybrid stat', 'hybrid nn']
            colors = ['b', 'g', 'seagreen']
            if not skip_L_models:
                if not only_L1:
                    areas = [no_hist_area, L0_area, L1_area, L2_area, hybrid_stat_area, hybrid_nn_area]
                    col_labels = ['no hist', 'L0', 'L1', 'L2', 'hybrid stat', 'hybrid nn']
                    colors = ['b', 'r', 'm', 'orange', 'g', 'seagreen']
                else:
                    areas = [no_hist_area, L1_area, hybrid_stat_area, hybrid_nn_area]
                    col_labels = ['no hist', 'L1', 'hybrid stat', 'hybrid nn']
                    colors = ['b', 'm', 'g', 'seagreen']


            cell_text = [(['%1.4f' % (area) for area in areas])]
            row_labels = ['area']

            table = plt.table(cellText=cell_text, rowLabels=row_labels, colColours=colors, colLabels=col_labels,
                              loc='top', cellLoc='center')
            table.scale(1, 1.2)
            plt.subplots_adjust(top=0.85)

            xs = [mono_no_hist_x, mono_hybrid_stat_x, mono_hybrid_nn_x]
            ys = [mono_no_hist_y, mono_hybrid_stat_y, mono_hybrid_nn_y]
            if not skip_L_models:
                if not only_L1:
                    xs = [mono_no_hist_x, mono_L0_x, mono_L1_x, mono_L2_x, mono_hybrid_stat_x, mono_hybrid_nn_x]
                    ys = [mono_no_hist_y, mono_L0_y, mono_L1_y, mono_L2_y, mono_hybrid_stat_y, mono_hybrid_nn_y]
                else:
                    xs = [mono_no_hist_x, mono_L1_x, mono_hybrid_stat_x, mono_hybrid_nn_x]
                    ys = [mono_no_hist_y, mono_L1_y, mono_hybrid_stat_y, mono_hybrid_nn_y]


            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                plt.fill_between(x, [h1_acc] * len(x), y, facecolor=colors[i], alpha=0.2)

            # plt.savefig(
            #     plots_dir + '\\by_hist_length\\len_' + str(history_len) + '_student_' + str(
            #         user_id) + '.png')
            # plt.savefig(
            #     plots_dir + '\\by_accuracy_range\\acc_' + '%.4f' % (auc_range,) + '_student_' + str(
            #         user_id) + '.png')
            # plt.savefig(
            #     plots_dir + '\\by_compatibility_range\\com_' + '%.4f' % (
            #     com_range,) + '_student_' + str(
            #         user_id) + '.png')
            # plt.savefig(
            #     plots_dir + '\\by_user_id\\student_' + str(user_id) + '_seed_' + str(seed) + '.png')

            plt.savefig(
                plots_dir + '\\plots\\student_' + str(user_id) + '.png')
            # plt.show()
            plt.clf()


def join_sim_and_auc():
    auc_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\chrono split\\1\\auc.csv'
    sim_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all features\\log.csv'
    out_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\e-learning\\sim_auc_correlation.csv'
    auc_df = pd.read_csv(auc_path)
    sim_df = pd.read_csv(sim_path)
    auc_df = auc_df.drop(columns=['train frac', 'instances', 'cos sim', 'train seed', 'comp range', 'acc range', 'h1 acc'])
    sim_by_users = sim_df.groupby('user_id')
    # del sim_df['user_id']
    with open(out_path, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(['user_id', 'correct_sim', 'skill_sim', 'opportunity_sim', 'original_sim', 'attempt_count_sim',
                         'tutor_mode_sim', 'answer_type_sim', 'type_sim', 'ms_first_response_sim', 'overlap_time_sim',
                         'no hist area', 'hybrid_stat area', 'L0 area', 'L1 area', 'L2 area'])
        for index, auc in auc_df.iterrows():
            sim = sim_by_users.get_group(auc['user_id']).reset_index(drop=True)
            # auc = auc.drop(columns=['user_id'])
            row = list(sim.loc[0]) + list(auc.values[1:])
            writer.writerow(row)


def safe_auc(x, y):
    x_reset = x.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    first_index = y_reset.first_valid_index()
    x_reset = list(x_reset[first_index:])
    y_reset = list(y_reset[first_index:])
    # make_monotonic(x_reset, y_reset)
    return auc(x_reset, y_reset)


def get_fixed_std(df_model, h1_mean_acc, users_h1_accs, weighted=False):
    users = df_model.groupby('user')
    model_mean_initial_acc = df_model.groupby('position').mean().reset_index(drop=True)['y'][0]
    df_fixed = pd.DataFrame(columns=['position', 'y'])
    for user_id, user_data in users:
        user_data = user_data.reset_index(drop=True)

        if weighted:
            size = int(user_data['size'][0])
            user_data = user_data.drop(columns=['size'])
            user_data = pd.concat([user_data] * size, ignore_index=True)

        user_y = user_data['y']
        model_user_initial_acc = user_y[0]
        h1_user_acc = users_h1_accs[user_id]
        y_min = min(min(user_y), h1_user_acc)
        y_max = max(max(user_y), h1_user_acc)
        if y_max == y_min:
        # if h1_user_acc >= model_user_initial_acc or y_max == y_min:
            continue
        fixed_y = (user_data['y'] - model_user_initial_acc) * ((model_mean_initial_acc - h1_mean_acc) / (y_max - y_min)) + model_mean_initial_acc
        df_fixed = df_fixed.append(pd.DataFrame({'position': user_data['position'], 'y': fixed_y}))
    return df_fixed.groupby('position').std()['y']


def plots_averaged_over_all_users(log_path, hybrid_log_path, results_dir,
                                  user_type='users', save_user_id='', simple_plots=False, weighted=True):

    model_names = [
        'L3',
        'hybrid',
        'no hist'
    ]
    labels_dict = {
        'no hist': 'without personalization',
        'L3': 'L3 personalization',
        'hybrid': 'hybrid personalization'
    }
    colors_dict = {
        'no hist': 'k',
        'L3': 'r',
        'hybrid': 'g'
    }
    labels = [labels_dict[i] for i in model_names]
    colors = [colors_dict[i] for i in model_names]

    log_df = pd.read_csv(log_path)
    hybrid_log_df = pd.read_csv(hybrid_log_path)

    log_by_users = log_df.groupby(['user_id'])
    hybrid_log_by_users = hybrid_log_df.groupby(['user_id'])

    user_ids = log_df.user_id.unique()

    final_df = pd.DataFrame(columns=['model', 'user', 'size', 'position', 'x', 'y'])

    user_count = 0
    for user_id in user_ids:
        user_count += 1
        print(str(user_count) + '/' + str(len(user_ids)) + ' user ' + str(user_id))

        user_log = log_by_users.get_group(user_id).reset_index(drop=True).groupby('diss weight').mean()
        user_hybrid_log = hybrid_log_by_users.get_group(user_id).reset_index(drop=True).groupby('std offset').mean()

        size = user_log['instances'][0]

        merged_df = pd.DataFrame(columns=['model', 'user', 'size', 'position', 'x', 'y'])
        merged_df = merged_df.append(
            pd.DataFrame({'model': 'h1', 'user':user_id, 'size':size, 'position':0, 'x': [user_log.loc[0]['no hist x']], 'y': [user_log.loc[0]['h1 acc']]}))
        for model_name in model_names:
            log = user_log
            positions = range(len(log))
            if model_name == 'hybrid':
                log = user_hybrid_log
                positions = reversed(range(len(log)))
            merged_df = merged_df.append(pd.DataFrame({'model': model_name, 'user': user_id, 'size':size, 'position': positions,
                                                       'x': log[model_name+' x'], 'y': log[model_name+' y']}))
        final_df = final_df.append(merged_df)

    groups = final_df.groupby('model')
    h1_group = groups.get_group('h1')[['size', 'y']].reset_index(drop=True)
    h1_acc_mean = np.average(h1_group['y'], weights=h1_group['size'])

    models_x = []
    models_y = []
    for model_name in model_names:
        group = groups.get_group(model_name)

        if weighted:
            weighted_group = group.copy()
            weighted_group['y'] = weighted_group['y'] * weighted_group['size']
            weighted_group['x'] = weighted_group['x'] * weighted_group['size']
            weighted_sum = weighted_group.groupby('position').sum()
            x = list(weighted_sum['x'] / weighted_sum['size'])
            y = list(weighted_sum['y'] / weighted_sum['size'])
        else:
            group_mean = group.groupby('position').mean()
            x = list(group_mean['x'])
            y = list(group_mean['y'])

        models_x += [x]
        models_y += [y]
    
    min_x = min(min(i) for i in models_x)
    max_x = max(max(i) for i in models_x)

    h1_x = [min_x, max_x]
    h1_y = [h1_acc_mean, h1_acc_mean]

    if simple_plots:
        linewidth = 12
        markersize = 40
        plot_alpha = 1

        for i in range(len(model_names)):
            plt.plot(models_x[i], models_y[i], colors[i], marker='.', label=labels[i], linewidth=linewidth, markersize=markersize, alpha=plot_alpha)
        plt.plot(h1_x, h1_y, 'k--', marker='.', label='before update', linewidth=linewidth, markersize=markersize, alpha=plot_alpha)

    else:
        for i in range(len(model_names)):
            plt.plot(models_x[i], models_y[i], colors[i], marker='.', label=labels[i])
        plt.plot(h1_x, h1_y, 'k--', marker='.', label='before update')

    if not simple_plots:
        xlabel = 'compatibility'
        ylabel = 'average accuracy'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='lower left')
        plt.grid()

        delimiter = ' = '
        if save_user_id == '':
            delimiter = ''
        title = 'Average tradeoff for %s%s%s' % (user_type, delimiter, save_user_id)
        plt.title(title)

        # plot std
        users_h1_accs = {}
        for user_id, user_data in groups.get_group('h1').groupby('user'):
            user_data = user_data.reset_index(drop=True)
            users_h1_accs[user_id] = user_data['y'][0]

        stds = [get_fixed_std(groups.get_group(i), h1_acc_mean, users_h1_accs, weighted) for i in model_names]
        for i in range(len(models_x)):
            x = models_x[i]
            y = models_y[i]
            std = stds[i]
            plt.fill_between(x, y+std, y-std, facecolor=colors[i], alpha=0.2)

    # else:
    #     size = final_df['size'].reset_index(drop=True)[0]
    #     fig = plt.gcf()
    #     # fig_size = fig.get_size_inches() * fig.dpi
    #     fig_size = fig.get_size_inches()
    #     plt.text(fig_size[0]/2, fig_size[1]/2, str(size), verticalalignment='top', horizontalalignment='center')

    delimiter = '_'
    if save_user_id == '':
        delimiter = ''
    file_name = 'average_plot_%s%s%s' % (user_type, delimiter, save_user_id)

    if simple_plots:
        plt.axis('off')
        plt.savefig(results_dir + '\\' + file_name + '.png', bbox_inches=0)
    else:
        plt.savefig(results_dir + '\\'+file_name+'.png')
    plt.show()
    plt.clf()


def sort_by(col):
    path_in = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\matific_encoded.csv'
    path_out = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\matific_sorted.csv'
    pd.read_csv(path_in).sort_values(by=[col]).to_csv(path_out, index=False)


def deal_with_empty():
    # path_in = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\matific_sorted.csv'
    # path_out = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\matific.csv'

    path_in = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\mallzee\\mallzee.csv'
    path_out = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\mallzee\\mallzee_filled.csv'

    df = pd.read_csv(path_in)
    print('rows='+str(len(df)))

    rows_to_drop = df[df['CurrentPrice'] == -1].index
    df_fixed = df.drop(rows_to_drop)
    # df_fixed = df.dropna()
    print('rows_without_nan=' + str(len(df_fixed)))
    
    df_fixed.to_csv(path_out, index=False)


def split_genres():
    movie_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\movies_metadata.csv'
    # user_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\ratings_small.csv'
    path_out = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\movies_splitted.csv'

    columns = ['budget', 'genres', 'id', 'original_language', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

    df_movies = pd.read_csv(movie_path)[columns]
    genres_col = df_movies['genres']
    df_movies = df_movies.drop(columns=['genres'])

    unique_genres = {}
    i = 0
    for genres_str in genres_col:
        i += 1
        print(str(i)+'/'+str(len(genres_col)))
        genres = ast.literal_eval(genres_str)
        for genre in genres:
            genre_name = genre['name']
            try:
                unique_genres[genre_name] = unique_genres[genre_name] + 1
            except KeyError:
                unique_genres[genre_name] = 1

    genres = []
    for genre in unique_genres.keys():
        if genre == 'Carousel Productions':
            break
        genres += [genre]

    df_genre_split = pd.DataFrame(columns=list(df_movies.columns) + genres)

    print('splitting')
    for i in range(len(df_movies)):
        print(str(i+1) + '/' + str(len(genres_col)))
        row_genres_str = ast.literal_eval(genres_col[i])
        row_genres = []
        for genre_str in row_genres_str:
            row_genres += [genre_str['name']]

        onehot = []
        for genre in genres:
            if genre in row_genres:
                onehot += [1]
            else:
                onehot += [0]
        df_genre_split.loc[i] = list(df_movies.loc[i]) + onehot

    df_genre_split.to_csv(path_out, index=False)


def join_movies_users():
    movie_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\movies_splitted.csv'
    user_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\ratings_small.csv'
    path_out = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\moviesKaggle.csv'

    df_movies = pd.read_csv(movie_path).astype({'movieId': 'int32'})
    df_users = pd.read_csv(user_path)
    df_joined = df_users.merge(df_movies, on='movieId', how='left').dropna()

    df_joined.to_csv(path_out, index=False)


def make_genres_count_column():
    path_in = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\moviesKaggle.csv'
    path_out = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\genresCount.csv'
    genres = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance',
              'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War',
              'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']
    df = pd.read_csv(path_in)
    with open(path_out, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(['userId', 'genre'])
        n = len(df)
        for i in range(n):
            print(str(i+1)+'/'+str(n))
            row = df.loc[i]
            user = row['userId']
            user_genres = row[genres]
            for j in range(len(user_genres)):
                name = user_genres.index[j]
                count = user_genres[j]
                if count == 1:
                    writer.writerow([user, name])

        # counts = df[genres].sum()
        # users = df['userId']
        # with open(path_out, 'w', newline='') as file_out:
        #     writer = csv.writer(file_out)
        #     writer.writerow(['genres'])
        #     for i in range(len(counts)):
        #         name = counts.index[i]
        #         count = counts[i]
        #         for j in range(count):
        #             writer.writerow([name])


def merge_all_user_csvs():
    dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\salaries\\80 split\\'
    user_types = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    df_log_merged = None
    df_hybrid_merged = None
    for user_type in user_types:
        user_type_dir = user_type + '\\'
        log_path = dataset + user_type_dir + 'log.csv'
        hybrid_log_path = dataset + user_type_dir + 'hybrid_log.csv'

        df_log = pd.read_csv(log_path)
        df_hybrid = pd.read_csv(hybrid_log_path)

        if df_log_merged is None:
            df_log_merged = pd.DataFrame(columns=df_log.columns)
            df_hybrid_merged = pd.DataFrame(columns=df_hybrid.columns)

        df_log_merged = df_log_merged.append(df_log)
        df_hybrid_merged = df_hybrid_merged.append(df_hybrid)

    df_log_merged.to_csv(dataset + '\\all\\log.csv', index=False)
    df_hybrid_merged.to_csv(dataset + '\\all\\hybrid_log.csv', index=False)


def merge_csv_parts(path, user_types):

    for user_type in user_types:
        user_type_dir = '%s\\%s' % (path, user_type)
        safe_make_dir(user_type_dir)
        df_log_merged = None
        df_hybrid_merged = None

        i = 1
        done = False
        while not done:
            part = 'part %d' % i
            i += 1
            try:
                log_path = '%s %s\\%s\\log.csv' % (path, part, user_type)
                hybrid_log_path = '%s %s\\%s\\hybrid_log.csv' % (path, part, user_type)

                df_log = pd.read_csv(log_path)
                df_hybrid = pd.read_csv(hybrid_log_path)

                if df_log_merged is None:
                    df_log_merged = pd.DataFrame(columns=df_log.columns)
                    df_hybrid_merged = pd.DataFrame(columns=df_hybrid.columns)

                df_log_merged = df_log_merged.append(df_log)
                df_hybrid_merged = df_hybrid_merged.append(df_hybrid)
            except FileNotFoundError:
                done = True

        df_log_merged.to_csv('%s\\log.csv' % (user_type_dir), index=False)
        df_hybrid_merged.to_csv('%s\\hybrid_log.csv' % (user_type_dir), index=False)


def split_users(log_path, hybrid_log_path, results_dir):
    df_log = pd.read_csv(log_path)
    df_hybrid_log = pd.read_csv(hybrid_log_path)
    log_by_users = df_log.groupby('user_id')
    hybrid_log_by_users = df_hybrid_log.groupby('user_id')

    for user_id, data in log_by_users:
        data.to_csv(results_dir + '%s_log.csv' % user_id)
    for user_id, data in hybrid_log_by_users:
        data.to_csv(results_dir + '%s_hybrid_log.csv' % user_id)


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_age_class_col():
    path_in = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\titanic\\titanic.csv'
    path_out = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\titanic\\titanic2.csv'
    df = pd.read_csv(path_in).fillna(value={'Age': 30})
    col = []
    age_classes = ['from 0 to 10', 'from 11 to 20', 'from 21 to 30', 'from 31 to 40', 'from 41 to 50', 'from 51 to 60',
                   'from 61 to 70', 'from 71 to 80']
    for i in df['Age']:
        if 0 <= i <= 10:
            j = 0
        if 11 <= i <= 20:
            j = 1
        if 21 <= i <= 30:
            j = 2
        if 31 <= i <= 40:
            j = 3
        if 41 <= i <= 50:
            j = 4
        if 51 <= i <= 60:
            j = 5
        if 61 <= i <= 70:
            j = 6
        if 71 <= i <= 80:
            j = 7
        col += [age_classes[j]]

    df['age_class'] = col
    df.to_csv(path_out)


def drop_rows_with_string():
    path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\breastCancer\\breastCancer'
    with open(path + ".csv", 'r', newline='') as file_in:
        with open(path + "_fixed.csv", 'w', newline='') as file_out:
            for line in file_in.readlines():
                if not '?' in line:
                    file_out.write(line)


# drop_rows_with_string()
# exit()

# def helping_kobi():
#     encoded = '╫£╫ס╫ש╫נ ╫נ╫ש╫á╫ר╫ע╫¿╫ª╫ש╫פ ╫ץ╫á╫ש╫¬╫ץ╫ק ╫₧╫ó╫¿╫¢╫ץ╫¬ ╫ס╫ó"╫₧'
#     decoded = 'חברת החשמל לישראל בעמ'
#     print(string)

plot = False
if plot:
    # log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\random split\\1\\merged_log.csv'
    # hybrid_stat_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\random split\\1 hybrid stat\\hybrid_log.csv'
    # hybrid_nn_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\random split\\1 hybrid nn\\regularization 0\\hybrid_log.csv'
    # plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\e-learning'

    log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\mallzee\\balanced\\merged\\merged_log.csv'
    hybrid_stat_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\mallzee\\balanced\\merged\\hybrid_stat_log.csv'
    hybrid_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\mallzee\\balanced\\merged\\hybrid_nn_log.csv'
    L1_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\mallzee\\balanced\\merged\\L1_log.csv'
    plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\mallzee'

    if not os.path.exists(plots_dir+'\\plots'):
        os.makedirs(plots_dir+'\\plots')

    area_calculator(log_path, hybrid_stat_log_path, hybrid_log_path, plots_dir, skip_L_models=True)
    # area_calculator(log_path, hybrid_stat_log_path, hybrid_nn_log_path, plots_dir, L1_log_path, only_L1=True, cut_by_min=True)

avg = True
if avg:
    # dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\salaries\\'
    # version = '80 split\\'
    # # user_types = ['all']
    # user_types = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    # dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\recividism\\'
    # version = '80 split [10] h1 500 h2 200 epochs\\'
    # user_types = ['race']
    # # user_types = ['race', 'sex', 'age_cat', 'c_charge_degree', 'score_text']

    # dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\titanic\\'
    # version = '80 split [] epochs h1 500 h2 800\\'
    # user_types = ['Pclass', 'Sex', 'AgeClass', 'Embarked']

    dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\assistment\\'
    version = '80 split [50]\\'
    user_types = ['user_id']

    individual_users = True
    only_split_users = False

    for user_type in user_types:
        user_type_dir = dataset + version + user_type + '\\'
        log_path = user_type_dir + 'log.csv'
        hybrid_log_path = user_type_dir + 'hybrid_log.csv'

        if only_split_users:
            by_user_id_dir = '%s\\log_by_user\\' % user_type_dir
            safe_make_dir(by_user_id_dir)
            split_users(log_path, hybrid_log_path, by_user_id_dir)
            continue

        if not individual_users:
            results_dir = dataset + version + 'averaged plots\\by user type'
            safe_make_dir(results_dir)
            plots_averaged_over_all_users(log_path, hybrid_log_path, results_dir,
                                          user_type=user_type)
        else:
            results_dir = dataset + version + 'averaged plots\\by user\\%s' % user_type
            safe_make_dir(results_dir)
            user_ids = pd.unique(pd.read_csv(log_path)['user_id'])
            for user_id in user_ids:
                user_log_path = '%slog_by_user\\%s_log.csv' % (user_type_dir, user_id)
                user_hybrid_log_path = '%slog_by_user\\%s_hybrid_log.csv' % (user_type_dir, user_id)
                plots_averaged_over_all_users(user_log_path, user_hybrid_log_path, results_dir,
                                              user_type=user_type, save_user_id=user_id, weighted=False,
                                              simple_plots=True)

# dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\titanic\\'
# version = '80 split [] epochs h1 500 h2 800'
# user_types = ['Pclass', 'Sex', 'AgeClass', 'Embarked']
#
# merge_csv_parts(dataset+version, user_types)

# log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\random split\\1\\merged_log.csv'
    # L1_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\random split\\1\\merged_log.csv'
    # hybrid_stat_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\random split\\1 hybrid stat\\hybrid_log.csv'
    # hybrid_nn_log_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plot archive\\ASSISTments_2010\\random split\\1 hybrid nn\\regularization 0\\hybrid_log.csv'
