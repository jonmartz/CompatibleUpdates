import json


def get_experiment_parameters(dataset_name, result_analysis=False):

    if dataset_name == 'assistment':
        # FOR MODEL TESTING
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
        min_hist_len_to_test = 0
        metrics = ['acc']

        # FOR RESULT ANALYSIS
        version = 'meta-learning'
        user_type = 'user_id'
        target_col = 'correct'
        experiment_type = 'large experiments'
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

    if dataset_name == 'citizen_science':
        # FOR MODEL TESTING
        # data settings
        target_col = 'd_label'
        original_categ_cols = []
        user_cols = ['user_id']
        skip_cols = []
        df_max_size = 0
        hists_already_determined = False
        # experiment settings
        train_frac = 0.7
        valid_frac = 0.2
        h1_len = 100
        h2_len = 10000000
        seeds = range(1)
        inner_seeds = range(1)
        weights_num = 5
        weights_range = [0, 1]
        # model settings
        model_type = 'tree'
        max_depth = None
        # ccp_alphas = [i/1000 for i in range(1, 10)] + [i/100 for i in range(1, 10)] + [i/10 for i in range(1, 10)]
        # ccp_alphas = [0.0001, 0.001, 0.01, 0.1]
        ccp_alphas = [0.01]
        ridge_alpha = 0.0001
        # user settings
        min_hist_len = 10
        max_hist_len = 10000000
        min_hist_len_to_test = 0
        metrics = ['acc']

        # FOR RESULT ANALYSIS
        version = 'large regularization'
        user_type = 'user_id'
        experiment_type = 'large experiments'
        performance_metric = 'acc'
        bin_size = 10

    if dataset_name == 'mooc':
        # FOR MODEL TESTING
        # data settings
        # target_col = 'urgency'
        target_col = 'confusion'
        # target_col = 'Opinion(1/0)'
        original_categ_cols = ['course_display_name', 'post_type', 'CourseType']
        user_cols = ['forum_uid']
        skip_cols = []
        # skip_cols = ['up_count', 'reads']
        df_max_size = 0
        hists_already_determined = False
        # experiment settings
        train_frac = 0.7
        valid_frac = 0.2
        h1_len = 50
        h2_len = 10000000
        seeds = [3]
        inner_seeds = range(1)
        weights_num = 30
        weights_range = [0, 1]
        # model settings
        model_type = 'tree'
        max_depth = None
        # ccp_alphas = [i/1000 for i in range(1, 10)]
        ccp_alphas = [0.0125]
        ridge_alpha = 0.0001
        # user settings
        min_hist_len = 10
        max_hist_len = 500
        min_hist_len_to_test = 120
        metrics = ['acc']

        # FOR RESULT ANALYSIS
        # version = 'meta-learning'
        # version = 'small test'
        version = 'best from valid'
        user_type = 'forum_uid'
        experiment_type = 'large experiments'
        performance_metric = 'acc'
        bin_size = 5

    if dataset_name == 'GZ':
        if not result_analysis:
            # data settings
            target_col = 'more_sessions'
            original_categ_cols = []
            user_cols = ['user']
            skip_cols = []
            df_max_size = 0
            hists_already_determined = False
            # experiment settings
            train_frac = 0.7
            valid_frac = 0.2
            h1_len = 50
            h2_len = 10000000
            seeds = range(3)
            inner_seeds = range(1)
            weights_num = 5
            weights_range = [0, 1]
            # model settings
            model_type = 'tree'
            max_depth = None
            # ccp_alphas = [i/100000 for i in range(1, 10)] + [i/10000 for i in range(1, 10)] + [i/1000 for i in range(1, 10)]
            ccp_alphas = [0.0, 0.0001, 0.01, 0.1]
            ridge_alpha = 0.0001
            # user settings
            min_hist_len = 10
            max_hist_len = 10000000
            min_hist_len_to_test = 0
            metrics = ['acc']
        else:
            version = 'small test'
            user_type = 'user'
            target_col = 'more_sessions'
            experiment_type = 'large experiments'
            performance_metric = 'acc'
            bin_size = 10

    if not result_analysis:
        return [target_col, original_categ_cols, user_cols, skip_cols, hists_already_determined, df_max_size,
                train_frac,
                valid_frac, h1_len, h2_len, seeds, inner_seeds, weights_num, weights_range, model_type, max_depth,
                ccp_alphas, ridge_alpha, min_hist_len, max_hist_len, metrics, min_hist_len_to_test]
    else:
        return version, user_type, target_col, experiment_type, performance_metric, bin_size, min_hist_len_to_test
