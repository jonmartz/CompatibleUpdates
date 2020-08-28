import json


def get_experiment_parameters(dataset_name, result_analysis=False):

    if dataset_name == 'assistment':
        if not result_analysis:
            # data settings
            target_col = 'correct'
            original_categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type']
            user_cols = ['user_id']
            # skip_cols = ['skill']
            skip_cols = []
            df_max_size = 0
            hists_already_determined = True
            # experiment settings
            train_frac = 0.7
            valid_frac = 0.2
            h1_len = 50
            h2_len = 50000
            seeds = range(5)
            inner_seeds = [0]
            weights_num = 10
            weights_range = [0, 1]
            # model settings
            model_type = 'tree'
            max_depth = None
            ccp_alpha = 0.005
            ridge_alpha = 0.0001
            # user settings
            min_hist_len = 50
            max_hist_len = 100000
            metrics = ['acc']
        else:
            version = 'meta-learning'
            user_type = 'user_id'
            target_col = 'correct'
            model_type = 'large experiments'
            performance_metric = 'acc'
            bin_size = 10

    if dataset_name == 'ednet':
        if not result_analysis:
            # data settings
            target_col = 'correct_answer'
            original_categ_cols = ['source', 'platform']
            user_cols = ['user']
            # skip_cols = []
            skip_cols = ['bkt_skill_learn_rate', 'bkt_skill_forget_rate', 'bkt_skill_guess_rate', 'bkt_skill_slip_rate']
            df_max_size = 100000
            hists_already_determined = False
            # experiment settings
            train_frac = 0.7
            valid_frac = 0.2
            h1_len = 20
            h2_len = 30000
            seeds = range(2)
            inner_seeds = range(2)
            weights_num = 3
            weights_range = [0, 1]
            # model settings
            max_depth = None
            ccp_alpha = 0.009
            ridge_alpha = 0.0001
            # user settings
            min_hist_len = 0
            max_hist_len = 100000
            metrics = ['acc', 'auc']
        else:
            version = 'unbalanced/20 users'
            user_type = 'user'
            target_col = 'correct_answer'
            model_type = 'simulated annealing'
            performance_metric = 'acc'
            bin_size = 10

    if dataset_name == 'salaries':
        if not result_analysis:
            # data settings
            target_col = 'salary'
            original_categ_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                   'sex',
                                   'native-country']
            user_cols = ['relationship']
            skip_cols = ['fnlgwt', 'education', 'native-country']
            df_max_size = 50000
            hists_already_determined = False
            # experiment settings
            train_frac = 0.7
            valid_frac = 0.2
            h1_len = 20
            h2_len = 50000
            seeds = [14]
            inner_seeds = range(1)
            weights_num = 20
            weights_range = [0, 1]
            # model settings
            model_type = 'tree'
            max_depth = None
            ccp_alpha = 0.008
            ridge_alpha = 0.0001
            # user settings
            min_hist_len = 50
            max_hist_len = 50000
            metrics = ['acc']
        else:
            version = 'meta-learning'
            user_type = 'relationship'
            target_col = 'salary'
            model_type = 'large experiments'
            performance_metric = 'acc'
            bin_size = 10

    if dataset_name == 'recividism':
        if not result_analysis:
            # data settings
            target_col = 'is_recid'
            original_categ_cols = ['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text', 'c_charge_desc']
            user_cols = ['race']
            # skip_cols = ['c_charge_desc', 'priors_count']
            skip_cols = ['score_text', 'age_cat']
            df_max_size = 0
            hists_already_determined = False
            # experiment settings
            train_frac = 0.7
            valid_frac = 0.2
            h1_len = 50
            h2_len = 20000
            seeds = [5]
            inner_seeds = range(1)
            weights_num = 20
            weights_range = [0, 1]
            # model settings
            model_type = 'tree'
            max_depth = None
            ccp_alpha = 0.005
            ridge_alpha = 0.0001
            # user settings
            min_hist_len = 0
            max_hist_len = 20000
            metrics = ['acc']
        else:
            version = 'meta-learning'
            user_type = 'race'
            target_col = 'is_recid'
            model_type = 'large experiments'
            performance_metric = 'acc'
            bin_size = 10

    # dataset_name = "mooc"
    # # data settings
    # target_col = 'Opinion(1/0)'
    # original_categ_cols = ['course_display_name', 'post_type', 'CourseType']
    # user_cols = ['forum_uid']
    # skip_cols = ['up_count', 'reads']
    # df_max_size = 100000
    # # experiment settings
    # train_frac = 0.8
    # valid_frac = 0.1
    # h1_len = 50
    # h2_len = 5000
    # seeds = range(1)
    # inner_seeds = range(2)
    # weights_num = 3
    # weights_range = [0, 1]
    # sim_ann_var = 0.05
    # max_sim_ann_iter = -1
    # iters_to_cooling = 100
    # # model settings
    # max_depth = None
    # ccp_alphas = [0.00001]
    # # user settings
    # min_hist_len = 50
    # max_hist_len = 2000
    # current_user_count = 0
    # users_to_not_test_on = []
    # only_these_users = []

    # dataset_name = "hospital_mortality"
    # # data settings
    # target_col = 'HOSPITAL_EXPIRE_FLAG'
    # original_categ_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']
    # # user_cols = ['MARITAL_STATUS']
    # user_cols = ['ADMISSION_TYPE', 'ETHNICITY']
    # skip_cols = []
    # df_max_size = 100000
    # # experiment settings
    # train_frac = 0.8
    # h1_len = 50
    # h2_len = 5000
    # seeds = range(30)
    # weights_num = 10
    # weights_range = [0, 1]
    # # model settings
    # max_depth = None
    # ccp_alpha = 0.001
    # # user settings
    # min_hist_len = 50
    # max_hist_len = 2000
    # current_user_count = 0
    # users_to_not_test_on = []
    # only_these_users = []

    if not result_analysis:
        return [target_col, original_categ_cols, user_cols, skip_cols, hists_already_determined, df_max_size,
                train_frac,
                valid_frac, h1_len, h2_len, seeds, inner_seeds, weights_num, weights_range, model_type, max_depth,
                ccp_alpha, ridge_alpha, min_hist_len, max_hist_len, metrics]
    else:
        return version, user_type, target_col, model_type, performance_metric, bin_size
