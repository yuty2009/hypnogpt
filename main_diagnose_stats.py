
import os
import csv
import yasa
import datetime
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from engine import sensitivity_specificity


sleep_datasets = {
    'cap_sleepedf' : {
        'data_dir' : 'e:/eegdata/sleep/cap_sleepedf/',
        'output_dir' : 'e:/eegdata/sleep/cap_sleepedf/output/',
    },
    'mnc' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/mnc/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/mnc/output/',
    },
    'isruc' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/isruc/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/isruc/output/',
    },
}

parser = argparse.ArgumentParser(description='Train and evaluate the Sleep Sequence Classification Model')
parser.add_argument('-D', '--dataset', default='cap_sleepedf', metavar='PATH',
                help='dataset used')
parser.add_argument('--pretrained_session', 
                    default='session_20240825135748',
                    metavar='str', help='session to pretrained model (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='xgboost',
                help='model architecture (default: gpt)')
parser.add_argument('--folds', default=10, type=int, metavar='N',
                    help='number of folds cross-valiation (default: 20)')
parser.add_argument('--start-fold', default=0, type=int, metavar='N',
                    help='manual fold number (useful on restarts)')
parser.add_argument('--splits', default='', type=str, metavar='PATH',
                    help='path to cross-validation splits file (default: none)')

def main():

    args = parser.parse_args()

    args.data_dir = sleep_datasets[args.dataset]['data_dir']
    args.output_dir = sleep_datasets[args.dataset]['output_dir']

    output_prefix = f"{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
    ann_f = open(os.path.join(args.data_dir, 'annotations.txt'), newline='')
    reader = csv.reader(ann_f)
    seqdata = [row[0:] for row in reader]
    # seqdata = crop_sleep_period(seqdata)
    sub_f = open(os.path.join(args.data_dir, 'subject_labels.txt'), newline='')
    reader = csv.reader(sub_f)
    labeldata = np.array(list(reader)).flatten()
    labeldata = [int(lb) for lb in labeldata]
    labeldata = np.asarray(labeldata)
    if args.dataset == 'cap_sleepedf':
        labeldata[labeldata > 0] = 1 # 0: normal, 1: abnormal  # cap
    elif args.dataset == 'mnc':
        labeldata[labeldata != 1] = 0 # 0: normal + hypersomnia, 1: narcolepsy

    print('Data for %d subjects has been loaded' % len(labeldata))
    num_subjects = len(labeldata)

    labels = np.array(labeldata, dtype=int)
    labels[labels > 0] = 1 # 0: normal, 1: abnormal  # cap
    # labels[labels != 1] = 0 # 0: normal + hypersomnia, 1: narcolepsy # mnc

    stage_names = ["W", "N1", "N2", "N3", "REM"]
    trans_names = [f"{stage_names[i]}->{stage_names[j]}" for i in range(5) for j in range(5)]

    data = []
    for i in range(num_subjects):
        hypno = [int(s) for s in seqdata[i]]
        data_stats = yasa.sleep_statistics(hypno, sf_hyp=1/30)
        counts, probs = transition_matrix(hypno)
        data_trans = probs.to_numpy().flatten()
        feature_cols = [*data_stats.keys(), *trans_names]
        data_sub = np.array(list(data_stats.values()) + list(data_trans))
        data.append(data_sub)
    data = np.vstack(data)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if args.start_fold <= 0:
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=0)
        splits_train, splits_test = [], []
        for (a, b) in kfold.split(np.arange(num_subjects)):
            splits_train.append(a)
            splits_test.append(b)
        np.savez(args.output_dir + '/splits.npz', splits_train=splits_train, splits_test=splits_test)
    else:
        splits = np.load(args.splits, allow_pickle=True)
        splits_train, splits_test = splits['splits_train'], splits['splits_test']

    params_xgb = {
        'learning_rate': 0.1,            # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
        'booster': 'gbtree',              # 提升方法，这里使用梯度提升树（Gradient Boosting Tree）
        'objective': 'binary:logistic',   # 损失函数，这里使用逻辑回归，用于二分类任务
        'max_leaves': 127,                # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
        'verbosity': 1,                   # 控制 XGBoost 输出信息的详细程度，0表示无输出，1表示输出进度信息
        'seed': 42,                       # 随机种子，用于重现模型的结果
        'nthread': -1,                    # 并行运算的线程数量，-1表示使用所有可用的CPU核心
        'colsample_bytree': 0.6,          # 每棵树随机选择的特征比例，用于增加模型的泛化能力
        'subsample': 0.7,                 # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
        'eval_metric': 'logloss',         # 评价指标，这里使用对数损失（logloss）
        'n_estimators': 20,
        'max_depth': 3,
    }

    # k-fold cross-validation
    train_accus, train_losses = np.zeros(args.folds), np.zeros(args.folds)
    test_accus,  test_losses  = np.zeros(args.folds), np.zeros(args.folds)
    test_cms, sensis, specis = np.zeros((args.folds, 2, 2)), np.zeros(args.folds), np.zeros(args.folds)
    test_ytrues, test_yprobs = [], []
    for fold in range(args.start_fold, args.folds):

        idx_train, idx_test = splits_train[fold], splits_test[fold]

        train_data = data[idx_train, :]
        train_labels = labels[idx_train]
        test_data = data[idx_test, :]
        test_labels = labels[idx_test]

        model = xgb.XGBClassifier(**params_xgb)
        model.fit(train_data, train_labels)

        savepath = os.path.join(args.output_dir, f"checkpoint/fold_{fold}/best.model")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        model.save_model(savepath)
        
        y_train = model.predict(train_data)
        y_prob = model.predict_proba(test_data)
        y_test = np.argmax(y_prob, axis=1)

        test_ytrues.append(test_labels)
        test_yprobs.append(y_prob)

        train_accus[fold] = balanced_accuracy_score(train_labels, y_train)
        train_losses[fold] = 0
        test_accus[fold] = balanced_accuracy_score(test_labels, y_test)
        test_losses[fold] = 0

        test_cms[fold] = confusion_matrix(test_labels, y_test)
        sensis[fold], specis[fold] = sensitivity_specificity(test_cms[fold])

    test_ytrues = np.concatenate(test_ytrues)
    test_yprobs = np.concatenate(test_yprobs)
    df_yy = pd.DataFrame({
        'y_true': test_ytrues,
        'y_prob': test_yprobs[:, 1],
    })
    df_yy.to_csv(os.path.join(args.output_dir, 'yy_xgboost.csv'))

    # Average over folds
    folds = [f"fold_{i}" for i in range(args.folds)] + ['average']
    train_accus = np.append(train_accus, np.mean(train_accus))
    train_losses = np.append(train_losses, np.mean(train_losses))
    test_accus  = np.append(test_accus, np.mean(test_accus))
    test_losses  = np.append(test_losses, np.mean(test_losses))
    sensis = np.append(sensis, np.mean(sensis))
    specis = np.append(specis, np.mean(specis))
    cm = np.sum(test_cms, axis=0)
    df_results = pd.DataFrame({
        'folds': folds,
        'train_accus': train_accus,
        'train_losses': train_losses,
        'test_accus' : test_accus,
        'test_losses' : test_losses,
        'test_sensis' : sensis,
        'test_specis' : specis,
    })
    df_results.to_csv(os.path.join(args.output_dir, 'results_xgboost.csv'))
    with open(os.path.join(args.output_dir, 'confusion_matrix.txt'), 'w') as f:
        csv.writer(f, delimiter=' ').writerows(cm)
        f.close()
    print(df_results)
    print(cm)


def transition_matrix(hypno):
    counts, probs = yasa.transition_matrix(hypno)
    if len(counts.columns) < 5:
        counts_comp = pd.DataFrame(np.zeros((5,5), dtype=int).tolist())
        probs_comp = pd.DataFrame(np.zeros((5,5), dtype=float).tolist())
        for r in counts.columns:
            for c in counts.columns:
                counts_comp.at[r, c] = counts.at[r, c]
                probs_comp.at[r, c] = probs.at[r, c]
        return counts_comp, probs_comp
    else:
        return counts, probs


if __name__ == "__main__":

    main()
