import pandas as pd

from modules.utils import generate_random_df

from modules.counterfactual_explainers.apply_dice import apply_dice

from modules.preprocessors.preprocess_german_credit_data import preprocess_german_credit_data, \
    german_credit_label_modifier
from modules.preprocessors.preprocess_adult_data import preprocess_adult_data, adult_label_modifier
from modules.preprocessors.preprocess_heart_data import preprocess_heart_data, heart_label_modifier
from modules.preprocessors.preprocess_diabetes_data import preprocess_diabetes_data, diabetes_label_modifier


def evaluate_counterfactual(ce: pd.DataFrame, reference: pd.DataFrame, labeler, decisive_columns,
                            criteria='both') -> bool:
    """
    method to evaluate a counterfactual explanation if the decision-making mechanism is known

    :param ce: counterfactual instance
    :param reference: original instance
    :param labeler: function to obtain the ground truth of the counterfactual instance
    :param decisive_columns: decisive columns for the ground truth decision
    :param criteria: 'both' criteria or criteria 'one' or 'two'
    :return: boolean if counterfactual explanation is true
    """

    # evaluate counterfactual explanations -- two criteria:
    # 1) if I would label this instance, would be belong to the desired class?
    criteria_1 = False
    if int(labeler(ce.drop(target, axis=1), seeds[i])[0]) == int(ce.loc[0, target]):
        criteria_1 = True

    # 2) are only the correct columns modified?
    diff = np.asarray(ce) - np.asarray(reference)
    diff_dict = {}
    for c, col in enumerate(list(ce.columns)):
        diff_dict[col] = diff[:, c]
    diff_df = pd.DataFrame.from_dict(diff_dict).drop(target, axis=1)

    criteria_2 = [False] * len(list(diff_df.columns))
    for c, col in enumerate(list(diff_df.columns)):
        if (diff_df.loc[0, col] != 0 and col in decisive_columns) or diff_df.loc[0, col] == 0:
            criteria_2[c] = True
    criteria_2 = all(criteria_2)

    if criteria == 'one':
        return criteria_1
    elif criteria == 'two':
        return criteria_2
    elif criteria == 'both':
        return criteria_1 and criteria_2
    else:
        raise ValueError('unknown criteria value')


if __name__ == '__main__':

    import os
    import argparse
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics.pairwise import euclidean_distances
    from raiutils.exceptions import UserConfigValidationException

    parser = argparse.ArgumentParser('execute experiments for TabBinCAIPI paper')

    parser.add_argument('-i', '--iterations', default='5',
                        help='experimental iterations')
    parser.add_argument('-ci', '--caipi_iterations', default='50',
                        help='caipi iterations')
    parser.add_argument('-d', '--data', default='data/german_credit_data.csv',
                        help='data set')
    parser.add_argument('-o', '--output', default='results/test.csv',
                        help='output file')
    parser.add_argument('-c', '--counterexamples', default='5',
                        help='number of counterexamples')
    parser.add_argument('-f', '--filtering', default='3.5',
                        help='filter counterexamples by median euclidean')

    args_dict = vars(parser.parse_args())

    output_path = str(args_dict['output'])
    if os.path.isfile(output_path):
        raise(FileExistsError('output file already exists.'))

    exp_iters = int(args_dict['iterations'])
    caipi_iterations = int(args_dict['caipi_iterations'])
    seeds = [42] * exp_iters
    for i in range(exp_iters):
        seeds[i] = seeds[i] * 10**i

    c = int(args_dict['counterexamples'])

    filtering = args_dict['filtering']
    if filtering == 'None':
        filtering = None
    else:
        filtering = float(filtering)

    df_path = str(args_dict['data'])
    if 'adult' in df_path:
        df_name = 'adult'
        target = 'income'
        preprocessor = preprocess_adult_data
        labeler = adult_label_modifier
        decisive_columns = ['workclass', 'educationalnum', 'hoursperweek']
    elif 'diabetes' in df_path:
        df_name = 'diabetes'
        target = 'diabetes'
        preprocessor = preprocess_diabetes_data
        labeler = diabetes_label_modifier
        decisive_columns = ['bloodpressure', 'bmi', 'glucose']
    elif 'credit' in df_path:
        df_name = 'credit'
        target = 'risk'
        preprocessor = preprocess_german_credit_data
        labeler = german_credit_label_modifier
        decisive_columns = ['housing', 'job', 'age', 'savingaccounts', 'checkingaccount']
    elif 'heart' in df_path:
        df_name = 'heart'
        target = 'disease'
        preprocessor = preprocess_heart_data
        labeler = heart_label_modifier
        decisive_columns = ['fbs', 'restecg', 'thal', 'chol', 'thalach']
    else:
        raise(ValueError('Unknown data set. Inspect data folder for valid data sets.'))

    print('### settings ####')
    print('experimental iterations: ' + str(exp_iters))
    print('caipi iterations: ' + str(caipi_iterations))
    print('random seeds: ' + str(seeds))
    print('data set: ' + str(df_name))
    print('')

    # initialize columns for output dataframe
    output_iterations = []
    output_caipi_iterations = []
    output_states = []
    output_fp_rates = []
    output_fn_rates = []
    output_precisions = []
    output_recalls = []
    output_accuracies = []
    output_corr_ces_pos = []
    output_corr_ces_neg = []
    output_labeled_size = []
    output_unlabeled_size = []

    for i in range(exp_iters):

        print('#### start experimental iteration ' + str(i+1) + ' ####')
        print()

        np.random.seed(seeds[i])

        # load data
        df_unlabeled, df_test, label_transformers, metric_transformers = preprocessor(df_path, test_ratio=0.3,
                                                                                      custom_mechanism=True,
                                                                                      seed=seeds[i])

        # initialize labeled data randomly
        df_labeled = generate_random_df(df_unlabeled, size=10, seed=seeds[i])

        # train kmeans
        # should be concat of unlabeled and labeled, but labeled is omitted here as it is purely random
        kmeans = KMeans(n_clusters=10, random_state=seeds[i])
        kmeans.fit(df_unlabeled.drop(target, axis=1))

        # start caipi optimization
        for j in range(caipi_iterations):

            print('#### caipi iteration ' + str(j + 1) + ' ####')
            print()

            output_iterations.append(i+1)
            output_caipi_iterations.append(j+1)

            # train classifier
            classifier = RandomForestClassifier(random_state=seeds[i], class_weight='balanced')
            classifier.fit(df_labeled.drop(target, axis=1), df_labeled[target])

            # select most-informative instance
            y_pred = classifier.predict_proba(df_unlabeled.drop(target, axis=1))
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            min_dist = min(abs(y_pred - 0.5))
            mii_index_continuous = np.where(abs(y_pred - 0.5) == min_dist)[0][0]
            mii = df_unlabeled.iloc[[mii_index_continuous]]
            mii_index = mii.index[0]

            # evaluate prediction
            y_pred = classifier.predict(mii.drop(target, axis=1))[0]
            if int(y_pred) == int(mii.loc[mii_index, target]):

                # generate and evaluate counterfactual explanation
                try:
                    ce = apply_dice(classifier, mii.drop(target, axis=1), df_unlabeled,
                                    list(mii.drop(target, axis=1).columns), target, number_cfs=1, seed=seeds[i])
                    corr_ce = evaluate_counterfactual(ce, mii, labeler, decisive_columns)
                except (TimeoutError, UserConfigValidationException, KeyError):
                    print('No counterfactual found, continue with next instance')
                    print()
                    corr_ce = False

                if corr_ce:
                    print('Right for the Right Reasons (RRR)')
                    print()

                    output_states.append('RRR')
                    df_labeled = pd.concat([df_labeled, mii])
                    df_unlabeled = df_unlabeled.drop(mii_index, axis=0)

                else:
                    print('Right for the Wrong Reasons (RWR)')
                    print()

                    output_states.append('RWR')

                    if c > 0:
                        # calculate median euclidean
                        df_unlabeled_sample = df_unlabeled.sample(100, random_state=seeds[i])
                        euclidean = euclidean_distances(mii.drop(target, axis=1), df_unlabeled_sample.drop(target, axis=1))
                        median_euclidean = np.median(euclidean)

                        print('median euclidean', median_euclidean)
                        print()

                        # generate counterexamples
                        cluster_pred = kmeans.predict(df_unlabeled.drop(target, axis=1))
                        cluster_pred_mii = kmeans.predict(mii.drop(target, axis=1))

                        cluster_index = np.where(cluster_pred == cluster_pred_mii)
                        counterexample_index = np.random.choice(cluster_index[0], c, replace=True)
                        df_dec = df_unlabeled.iloc[counterexample_index][decisive_columns
                                                                         + [target]].reset_index(drop=True)

                        df_no_dec = generate_random_df(df_unlabeled.drop(decisive_columns + [target], axis=1), c)
                        df_counterexample = pd.concat([df_no_dec, df_dec], axis=1)

                        if filtering:
                            if median_euclidean <= filtering:
                                df_labeled = pd.concat([df_labeled, mii, df_counterexample])
                                df_unlabeled = df_unlabeled.drop(mii_index, axis=0)
                            else:
                                df_labeled = pd.concat([df_labeled, mii])
                                df_unlabeled = df_unlabeled.drop(mii_index, axis=0)

                        else:
                            df_labeled = pd.concat([df_labeled, mii, df_counterexample])
                            df_unlabeled = df_unlabeled.drop(mii_index, axis=0)

                    else:
                        df_labeled = pd.concat([df_labeled, mii])
                        df_unlabeled = df_unlabeled.drop(mii_index, axis=0)

            else:
                print('Wrong (W)')
                print()

                output_states.append('W')
                df_labeled = pd.concat([df_labeled, mii])
                df_unlabeled = df_unlabeled.drop(mii_index, axis=0)

            # evaluate predictive quality
            y_pred = classifier.predict(df_test.drop(target, axis=1))
            tn, fp, fn, tp = confusion_matrix(df_test[target], y_pred).ravel()
            fp_rate = fp / (fp + tn)
            fn_rate = fn / (fn + tp)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / len(y_pred)

            print('fp rate:', fp_rate)
            print('fn rate:', fn_rate)
            print('precision:', precision)
            print('recall:', recall)
            print('accuracy:', accuracy)
            print()

            output_fp_rates.append(fp_rate)
            output_fn_rates.append(fn_rate)
            output_precisions.append(precision)
            output_recalls.append(recall)
            output_accuracies.append(accuracy)

            # evaluate explanatory quality for each correctly predicted test instance
            df_eval_ces = df_test.copy()
            df_eval_ces['preds'] = y_pred
            df_eval_ces = df_eval_ces[df_eval_ces[target] == df_eval_ces['preds']]

            if len(df_eval_ces) > 100:
                select_index = np.random.choice(list(range(0, len(df_eval_ces))), 100, replace=False)
                df_eval_ces = df_eval_ces.iloc[select_index]

            corr_ces = [False] * len(df_eval_ces)
            for n in range(len(df_eval_ces)):

                # generate counterfactual explanation
                try:
                    ce = apply_dice(classifier, df_eval_ces.iloc[[n]].drop([target, 'preds'], axis=1), df_unlabeled,
                                    list(df_eval_ces.drop([target, 'preds'], axis=1).columns), target,
                                    number_cfs=1, seed=seeds[i])
                except (TimeoutError, UserConfigValidationException, KeyError):
                    print('No counterfactual found, continue with next instance')
                    print()
                    continue

                # evaluate counterfactual explanation
                corr_ces[n] = evaluate_counterfactual(ce, df_eval_ces.iloc[[n]].drop('preds', axis=1),
                                                      labeler, decisive_columns)

            # get ratio of correct counterfactuals conditioned on class
            df_eval_ces['corr_ces'] = corr_ces

            try:
                corr_ces_pos = len(df_eval_ces[(df_eval_ces[target] == 1) & (df_eval_ces['corr_ces'] == True)]) \
                    / len(df_eval_ces[df_eval_ces[target] == 1])
            except ZeroDivisionError:
                corr_ces_pos = 0
            try:
                corr_ces_neg = len(df_eval_ces[(df_eval_ces[target] == 0) & (df_eval_ces['corr_ces'] == True)]) \
                    / len(df_eval_ces[df_eval_ces[target] == 0])
            except ZeroDivisionError:
                corr_ces_neg = 0

            print('correct ces positive class:', corr_ces_pos,
                  '(n='+str(len(df_eval_ces[df_eval_ces[target] == 1]))+str(')'))
            print('correct ces negative class:', corr_ces_neg,
                  '(n='+str(len(df_eval_ces[df_eval_ces[target] == 0]))+str(')'))
            print()
            print('labeled size:', len(df_labeled))
            print('unlabeled size:', len(df_unlabeled))
            print()

            output_corr_ces_pos.append(corr_ces_pos)
            output_corr_ces_neg.append(corr_ces_neg)
            output_labeled_size.append(len(df_labeled))
            output_unlabeled_size.append(len(df_unlabeled))

        print('#### end experimental iteration ' + str(i+1) + ' ####')
        print()

    output_df = pd.DataFrame()
    output_df['exp_iter'] = output_iterations
    output_df['caipi_iter'] = output_caipi_iterations
    output_df['state'] = output_states
    output_df['fp_rate'] = output_fp_rates
    output_df['fn_rates'] = output_fn_rates
    output_df['precision'] = output_precisions
    output_df['recall'] = output_recalls
    output_df['accuracy'] = output_accuracies
    output_df['corr_ces_pos'] = output_corr_ces_pos
    output_df['corr_ces_neg'] = output_corr_ces_neg
    output_df['labeled_size'] = output_labeled_size
    output_df['unlabeled_size'] = output_unlabeled_size
    output_df.to_csv(output_path)
