import csv
import numpy as np
import os
import pandas as pd
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


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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

