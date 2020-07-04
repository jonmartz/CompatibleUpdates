import numpy as np
import pandas as pd
import json
import os
import csv


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_ednet_content(dataset_dir):
    questions = pd.read_csv('%s/contents/questions.csv' % dataset_dir, index_col='question_id')
    lectures = pd.read_csv('%s/contents/lectures.csv' % dataset_dir, index_col='lecture_id')
    skills_from_questions = pd.read_csv('%s/contents/question_skills.csv' % dataset_dir, index_col='skill_id')
    skills_from_lectures = pd.read_csv('%s/contents/lecture_skills.csv' % dataset_dir, index_col='skill_id')
    return questions, lectures, skills_from_questions, skills_from_lectures


def write_bkt_matrix(users_dir, users, dataset_dir):
    questions, lectures, skills_from_questions, skills_from_lectures = get_ednet_content(dataset_dir)
    data_transposed = []
    starts = []
    lengths = []
    resources = []
    user_idx = 1
    for user in users:
        print('user %d/%d' % (user_idx, len(users)))
        user_idx += 1
        starts.append(len(data_transposed) + 1)
        rows = pd.read_csv('%s/u%d.csv' % (users_dir, user), usecols=['action_type', 'item_id', 'user_answer']).values
        questions_in_bundle = {}
        for row in rows:
            action = row[0]
            if action == 'respond':
                question_id = row[1]
                question = questions.loc[question_id]
                answer = row[2]
                correct_answer = answer == question['correct_answer']
                if correct_answer:
                    questions_in_bundle[question_id] = 2
                else:
                    questions_in_bundle[question_id] = 1
            elif action == 'submit':
                for question_id, answer in questions_in_bundle.items():
                    question = questions.loc[question_id]
                    skill_ids = [int(i) for i in question['tags'].split(';')]
                    skill_indexes = skills_from_questions.loc[skill_ids]['skill_index']
                    data_col = [0] * len(skills_from_questions)
                    for skill_index in skill_indexes:
                        data_col[skill_index] = answer
                    data_transposed.append(data_col)
                    resources.append(1)
                questions_in_bundle = {}
            elif action == 'quit' and row[1][0] == 'l':
                lecture_id = row[1]
                lecture = lectures.loc[lecture_id]
                skill_id = lecture['tags']
                skill_index = int(skills_from_lectures.loc[skill_id]['skill_index'])
                data_transposed.append([0] * len(skills_from_questions))
                resources.append(skill_index + 1)
        lengths.append(len(data_transposed) - (starts[-1] - 1))
    output = {'data': np.array(data_transposed).T.tolist(), 'starts': starts, 'lengths': lengths,
              'resources': resources,
              'num_resources': len(skills_from_lectures)}
    with open('%s/bkt_data.json' % dataset_dir, 'w') as file:
        json.dump(output, file)


def write_error_rates(users_dir, users, dataset_dir):
    questions, lectures, skills_from_questions, skills_from_lectures = get_ednet_content(dataset_dir)
    skills_answers, bundles_answers, questions_answers = {}, {}, {}
    for user_idx, user in enumerate(users):
        print('user %d/%d' % ((user_idx + 1), len(users)))
        df_user = pd.read_csv('%s/u%d.csv' % (users_dir, user))
        questions_in_bundle = {}
        for index, row in df_user.iterrows():
            action = row['action_type']
            if action == 'respond':
                question_id = row['item_id']
                question = questions.loc[question_id]
                answer = row['user_answer']
                correct_answer = answer == question['correct_answer']
                if correct_answer:
                    questions_in_bundle[question_id] = 1
                else:
                    questions_in_bundle[question_id] = 0
            elif action == 'submit':
                if len(questions_in_bundle) > 0:
                    bundle_id = row['item_id']
                    bundle_answers = get_or_init_value_from_dict(bundles_answers, bundle_id, [])
                    for question_id, answer in questions_in_bundle.items():
                        bundle_answers.append(answer)
                        question_answers = get_or_init_value_from_dict(questions_answers, question_id, [])
                        question_answers.append(answer)
                        question = questions.loc[question_id]
                        for skill_id in [int(i) for i in question['tags'].split(';')]:
                            skill_answers = get_or_init_value_from_dict(skills_answers, skill_id, [])
                            skill_answers.append(answer)
                questions_in_bundle = {}
    write_item_error_rates('skill', skills_answers, dataset_dir)
    write_item_error_rates('bundle', bundles_answers, dataset_dir)
    write_item_error_rates('question', questions_answers, dataset_dir)


def get_or_init_value_from_dict(dictionary, key, initial_structure):
    if key in dictionary:
        value = dictionary[key]
    else:
        value = initial_structure
        dictionary[key] = value
    return value


def write_item_error_rates(item_type, item_type_answers, dataset_dir):
    item_ids, error_rates, correct_answers, wrong_answers = [], [], [], []
    for item_id, answers in item_type_answers.items():
        if len(answers) == 0:
            print('empty value')
        item_ids.append(item_id)
        error_rates.append(1 - np.mean(answers))
        sum_answers = sum(answers)
        correct_answers.append(sum_answers)
        wrong_answers.append(len(answers) - sum_answers)
    out_path = '%s/contents/error rates/%s_error_rates.csv' % (dataset_dir, item_type)
    pd.DataFrame({'item_id': item_ids, 'error_rate': error_rates, 'correct_answers': correct_answers,
                  'wrong_answers': wrong_answers}).to_csv(out_path, index=False)


def extract_features(users_dir, users, dataset_dir):
    # skill_error_rates = pd.read_csv('%s/contents/error rates/skill_error_rates.csv' % dataset_dir,
    #                                 index_col='item_id')
    # bundle_error_rates = pd.read_csv('%s/contents/error rates/bundle_error_rates.csv' % dataset_dir,
    #                                  index_col='item_id')
    # question_error_rates = pd.read_csv('%s/contents/error rates/question_error_rates.csv' % dataset_dir,
    #                                    index_col='item_id')
    # with open('%s/bkt_output.json' % dataset_dir, 'r') as file:
    #     bkt_output = json.load(file)
    # bkt_learns = bkt_output['learns']
    # bkt_forgets = bkt_output['forgets']
    # bkt_guesses = bkt_output['guesses']
    # bkt_slips = bkt_output['slips']
    questions, lectures, skills_from_questions, skills_from_lectures = get_ednet_content(dataset_dir)
    bundles = pd.read_csv('%s/contents/bundles.csv' % dataset_dir, index_col='bundle_id')

    with open('%s/ednet.csv' % dataset_dir, 'w', newline='') as file_out:
        header = [
            'user',
            'source',
            'platform',
            # 'skill_error_rate',
            # 'bundle_error_rate',
            # 'question_error_rate',
            'question_part',
            'user_skill_error_rate',
            'user_skill_opportunities',
            'user_skill_explanations',
            'user_skill_explanation_time',
            'user_skill_lectures',
            'user_skill_lecture_time',
            'user_bundle_error_rate',
            'user_bundle_opportunities',
            'user_bundle_explanations',
            'user_bundle_explanation_time',
            'user_question_error_rate',
            'user_question_opportunities',
            'time_to_answer',
            'num_changes_of_answer',
            # 'bkt_skill_learn_rate',
            # 'bkt_skill_forget_rate',
            # 'bkt_skill_guess_rate',
            # 'bkt_skill_slip_rate',
            'correct_answer',
        ]
        writer = csv.writer(file_out)
        writer.writerow(header)
        enter_time = 0
        last_time = 0
        for user_idx, user in enumerate(users):
            print('user %d/%d' % (user_idx + 1, len(users)))
            df_user = pd.read_csv('%s/u%d.csv' % (users_dir, user))
            user_skills, user_bundles, user_questions = {}, {}, {}
            questions_in_bundle = {}
            for index, row in df_user.iterrows():
                action = row['action_type']

                if action == 'enter':  # entered either bundle, explanation or lecture
                    enter_time = row['timestamp']
                    last_time = enter_time

                elif action == 'respond':
                    question_id = row['item_id']
                    question = questions.loc[question_id]
                    is_correct = int(row['user_answer'] == question['correct_answer'])
                    question_in_bundle = get_or_init_value_from_dict(
                        questions_in_bundle, question_id, {'answers': [], 'times': []})
                    question_in_bundle['answers'].append(is_correct)
                    question_in_bundle['times'].append(row['timestamp'] - last_time)
                    last_time = row['timestamp']

                elif action == 'submit':
                    if len(questions_in_bundle) > 0:
                        bundle_id = row['item_id']
                        user_bundle = get_or_init_value_from_dict(
                            user_bundles, bundle_id, {'error_rate': [1], 'explanations': []})
                        for question_id, question_in_bundle in questions_in_bundle.items():
                            is_correct = question_in_bundle['answers'][-1]
                            user_question_answers = get_or_init_value_from_dict(user_questions, question_id, [0])
                            question = questions.loc[question_id]
                            skill_ids = [int(i) for i in question['tags'].split(';')]
                            question_skills = []
                            for skill_id in skill_ids:
                                user_skill = get_or_init_value_from_dict(
                                    user_skills, skill_id, {'answers': [0], 'explanations': [], 'lectures': []})
                                question_skills.append(user_skill)

                            # # BKT features
                            # question_skill_indexes = list(skills_from_questions.loc[skill_ids]['skill_index'])
                            # lecture_skill_indexes = list(skills_from_lectures.loc[skill_ids]['skill_index'].dropna())
                            # learn = [bkt_learns[int(i)] for i in lecture_skill_indexes]
                            # forget = [bkt_forgets[int(i)] for i in lecture_skill_indexes]
                            # guess = [bkt_guesses[i] for i in question_skill_indexes]
                            # slips = [bkt_slips[i] for i in question_skill_indexes]

                            user_skill_error_rate = np.mean([1 - np.mean(i['answers']) for i in question_skills])
                            if np.isnan(user_skill_error_rate):
                                raise ValueError('mean of empty slice')

                            writer.writerow([
                                user,  # user
                                row['source'],  # source
                                row['platform'],  # platform
                                # np.mean(skill_error_rates.loc[skill_ids]['error_rate']),  # skill_error_rate
                                # bundle_error_rates.loc[bundle_id]['error_rate'],  # bundle_error_rate
                                # question_error_rates.loc[question_id]['error_rate'],  # question_error_rate
                                question['part'],  # question_part
                                user_skill_error_rate,  # user_skill_error_rate
                                sum([len(i['answers']) - 1 for i in question_skills]),  # user_skill_opportunities
                                sum([len(i['explanations']) for i in question_skills]),  # user_skill_explanations
                                sum([sum(i['explanations']) for i in question_skills]),  # user_skill_explanation_time
                                sum([len(i['lectures']) for i in question_skills]),  # user_skill_lectures
                                sum([sum(i['lectures']) for i in question_skills]),  # user_skill_lecture_time
                                np.mean(user_bundle['error_rate']),  # user_bundle_error_rate
                                len(user_bundle['error_rate']) - 1,  # user_bundle_opportunities
                                len(user_bundle['explanations']),  # user_bundle_explanations
                                sum(user_bundle['explanations']),  # user_bundle_explanation_time
                                1 - np.mean(user_question_answers),  # user_question_error_rate
                                len(user_question_answers) - 1,  # user_question_opportunities
                                sum(question_in_bundle['times']),  # time_to_answer
                                len(question_in_bundle['times']) - 1,  # num_changes_of_answer
                                # np.mean(learn),  # bkt_skill_learn_rate
                                # np.mean(forget),  # bkt_skill_learn_rate
                                # np.mean(guess),  # bkt_skill_learn_rate
                                # np.mean(slips),  # bkt_skill_learn_rate
                                is_correct,  # target
                            ])
                            # append for future error rates
                            if len(user_question_answers) == 1:
                                user_question_answers[0] = is_correct
                            else:
                                user_question_answers.append(is_correct)
                            for question_skill in question_skills:
                                if len(question_skill['answers']) == 1:
                                    question_skill['answers'][0] = is_correct
                                else:
                                    question_skill['answers'].append(is_correct)
                        # append for future error rates
                        user_bundle_error_rate = [q['answers'][-1] for i, q in questions_in_bundle.items()]
                        if len(user_bundle_error_rate) == 0:
                            raise ValueError('mean of empty slice')
                        if len(user_bundle['error_rate']) == 1:
                            user_bundle['error_rate'][0] = 1 - np.mean(user_bundle_error_rate)
                        else:
                            user_bundle['error_rate'].append(1 - np.mean(user_bundle_error_rate))

                    questions_in_bundle = {}

                elif action == 'quit':
                    time_in_item = row['timestamp'] - enter_time
                    item_id = row['item_id']
                    item_type = item_id[0]
                    if item_type == 'e':  # quit explanation
                        user_bundle = get_or_init_value_from_dict(
                            user_bundles, bundle_id, {'error_rate': [], 'explanations': []})
                        user_bundle['explanations'].append(time_in_item)
                        skill_ids = [int(i) for i in bundles.loc[int(item_id[1:])]['tags'].split(';')]
                        for skill_id in skill_ids:
                            user_skill = get_or_init_value_from_dict(
                                user_skills, skill_id, {'answers': [0], 'explanations': [], 'lectures': []})
                            user_skill['explanations'].append(time_in_item)
                    elif item_type == 'l':  # quit lecture
                        skill_id = lectures.loc[item_id]['tags']
                        user_skill = get_or_init_value_from_dict(
                            user_skills, skill_id, {'answers': [0], 'explanations': [], 'lectures': []})
                        user_skill['lectures'].append(time_in_item)


def write_bundle_content(dataset_dir):
    questions, lectures, skills_from_questions, skills_from_lectures = get_ednet_content(dataset_dir)
    bundles = {}
    for index, question in questions.iterrows():
        bundle = get_or_init_value_from_dict(bundles, question['bundle_id'], set())
        skill_ids = [int(i) for i in question['tags'].split(';')]
        for skill_id in skill_ids:
            bundle.add(skill_id)
    with open('%s/contents/bundles.csv' % dataset_dir, 'w', newline='') as file_out:
        header = ['bundle_id', 'tags']
        writer = csv.writer(file_out)
        writer.writerow(header)
        for bundle_id, tags in bundles.items():
            writer.writerow([bundle_id[1:], ';'.join([str(i) for i in sorted(tags)])])


dataset_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/ednet'
users_dir = '%s/KT3' % dataset_dir

# users = [3, 6]
# lens: [5000    4500    4000    3500    3000    2500    2000    1500    1000    500
# KBs:  [205     185     165     145     125     105     85      65      45      25
users = [
    562650,  # 10000, 405kb
    549078,  # 9500, 385kb
    344182,  # 9000, 365kb
    310618,  # 8500, 345kb
    577484,  # 8000, 325kb
    335760,  # 7500, 305kb
    274205,  # 7000, 285kb
    656267,  # 6500, 265kb
    640295,  # 6000, 245kb
    1655,    # 5500, 225kb
    503066,  # 5000, 205kb
    480292,  # 4500, 185kb
    295872,  # 4000, 165kb
    357194,  # 3500, 145kb
    759070,  # 3000, 125kb
    332183,  # 2500, 105kb
    342307,  # 2000, 85kb
    357889,  # 1500, 65kb
    773623,  # 1000, 45kb
    533110,  # 500, 25kb
]
# write_bkt_matrix(users_dir, users, dataset_dir)
# write_error_rates(users_dir, users, dataset_dir)
extract_features(users_dir, users, dataset_dir)

print('done')
