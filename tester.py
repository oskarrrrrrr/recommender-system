#!/usr/bin/env python
# coding: utf-8

import attr
from abc import ABC, abstractmethod

from surprise import Dataset, Reader
from surprise import SVD, SVDpp, KNNBasic

import scipy.stats
import pandas as pd
import numpy as np
import datetime as dt
from timeit import default_timer as timer

pd.options.mode.chained_assignment = None  # default='warn'


@attr.s
class TestCase:
    alg = attr.ib(kw_only=True)
    args = attr.ib(kw_only=True, factory=dict)
    alg_name = attr.ib(kw_only=True)
    filename = attr.ib(kw_only=True)
    test_name = attr.ib(kw_only=True, default="")
    test_description = attr.ib(kw_only=True, default="")

    @alg_name.default
    def _get_alg_name_form_alg(self):
        return type(self.alg).__name__


class TrainTestSplitProvider(ABC):
    def __init__(self):
        self._df = None

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass


class CrossValidTrainTestProvider(TrainTestSplitProvider):
    def __init__(self, *, splits=3):
        super().__init__()
        self.splits = splits

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self._df = new_df

        # shuffle
        self._df = self._df.sample(frac=1).reset_index(drop=True)

        self.borders = np.arange(0, self.splits + 1) * len(self._df) / self.splits
        self.borders = self.borders.astype(np.int64)
        self.borders[-1] = self._df.shape[0]

        self.right_i = 1

    def __next__(self):
        while self.right_i < len(self.borders):
            test_left_i = self.borders[self.right_i - 1]
            test_right_i = self.borders[self.right_i]
            test_df = self._df[test_left_i:test_right_i]

            train_head_df = self._df[0 : self.borders[self.right_i - 1]]
            train_tail_df = self._df[self.borders[self.right_i] : self.borders[-1]]
            train_df = train_head_df.append(train_tail_df)

            self.right_i += 1
            return train_df, test_df
        raise StopIteration


class NaiveTailTrainTestProvider(TrainTestSplitProvider):
    def __init__(self, *, tail_len):
        super().__init__()
        self.tail_len = tail_len
        self.provide_split = True

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self._df = new_df
        self._df.sort_values(by=["timestamp"], inplace=True, ascending=True)

    def __next__(self):
        if self.provide_split:
            self.provide_split = False
            return self._df[: -self.tail_len], self._df[-self.tail_len :]
        self.provide_split = True
        raise StopIteration


class ColdStartTrainTestProvider(TrainTestSplitProvider):
    def __init__(self, *, users_num, head_len, sort_by_date=True):
        super().__init__()
        self.users_num = users_num
        self.head_len = head_len
        self.sort_by_date = sort_by_date
        self.provide_split = True

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self._df = new_df
        if self.sort_by_date:
            self._df.sort_values(by=["timestamp"], inplace=True, ascending=True)
        else:
            # shuffle
            self._df = self._df.sample(frac=1).reset_index(drop=True)

    def __next__(self):
        if self.provide_split:
            self.provide_split = False
            unique_users = self._df["userId"].unique()
            sample_users_idx = np.random.choice(
                len(unique_users), replace=False, size=self.users_num
            )
            sample_users = unique_users[sample_users_idx]
            self._df["user_rating_num"] = self._df.groupby("userId").cumcount()
            train_idx = (self._df["user_rating_num"] <= self.head_len) | ~(
                self._df["userId"].isin(sample_users)
            )
            test_idx = (self._df["user_rating_num"] > self.head_len) & (
                self._df["userId"].isin(sample_users)
            )
            train_df = self._df[train_idx]
            test_df = self._df[test_idx]
            return train_df, test_df

        self.provide_split = True
        raise StopIteration


def rmse(targets, preds):
    return np.sqrt(((preds - targets) ** 2).mean())


def mse(targets, preds):
    return ((preds - targets) ** 2).mean()


def mae(targets, preds):
    return (np.abs(preds - targets)).mean()


def precision(targets, preds):
    return (targets == np.round(preds)).sum() / len(targets)


def rmse_4_5(targets, preds):
    use = (targets >= 4) | (preds >= 4)
    return rmse(targets[use], preds[use])


def mae_4_5(targets, preds):
    use = (targets >= 4) | (preds >= 4)
    return mae(targets[use], preds[use])


def mae_4_5_unsym(targets, preds):
    only_pred_is_4_or_5 = (preds >= 4) & (targets < 4)
    if only_pred_is_4_or_5.sum():
        false_recommendation_penalty = mae(
            targets[only_pred_is_4_or_5], preds[only_pred_is_4_or_5]
        )
    else:
        false_recommendation_penalty = 0.0

    only_target_is_4_or_5 = (targets >= 4) & (preds < 4)
    if only_target_is_4_or_5.sum():
        missed_recommendation_penalty = mae(
            targets[only_target_is_4_or_5], preds[only_target_is_4_or_5]
        )
    else:
        missed_recommendation_penalty = 0

    both_are_4_5 = (targets >= 4) & (preds >= 4)
    if both_are_4_5.sum():
        good_recommendation_penalty = mae(targets[both_are_4_5], preds[both_are_4_5])
    else:
        good_recommendation_penalty = 0

    return (
        false_recommendation_penalty * 1
        + missed_recommendation_penalty * 0.5
        + good_recommendation_penalty * 0.5
    ) / 2


def spearman(targets, preds, max_n=20):
    s = np.argsort(preds[::-1])
    targets = targets[s][:max_n]
    preds = preds[s][:max_n]
    return scipy.stats.spearmanr(targets, preds)[0]


def kendalltau(targets, preds, max_n=20):
    s = np.argsort(preds[::-1])
    targets = targets[s][:max_n]
    preds = preds[s][:max_n]
    return scipy.stats.kendalltau(targets, preds)[0]

def top_n_average_rating(target, preds, max_n=20):
    s = np.argsort(preds[::-1])
    preds = preds[s][:max_n]
    return preds.mean()


all_score_functions = dict(
    rmse=rmse,
    mse=mse,
    mae=mae,
    precision=precision,
    rmse_4_5=rmse_4_5,
    mae_4_5=mae_4_5,
    mae_4_5_unsym=mae_4_5_unsym,
)


@attr.s
class Tester:
    train_test_iter = attr.ib(kw_only=True, default=CrossValidTrainTestProvider())
    score_functions = attr.ib(kw_only=True)
    df = attr.ib(kw_only=True)
    df2ds = attr.ib(kw_only=True)
    save_results = attr.ib(kw_only=True, default=True)
    verbose = attr.ib(kw_only=True, default=True)

    def _introduce_test(self, n_tests):
        self.vprint("Starting tests...")
        self.vprint(f"Number of tests: {n_tests}")
        self.vprint(f"Splitter: {type(self.train_test_iter).__name__}")
        self.vprint(f'Scoring functions: {", ".join(self.score_functions.keys())}')
        if self.save_results:
            self.vprint("Results will be saved to files.")

    def test(self, test_cases):
        self.vprint = print if self.verbose else lambda *a, **k: None
        self._introduce_test(len(test_cases))
        self.vprint("." * len(test_cases))
        self.first_test = True
        for self.test_num, test_case in enumerate(test_cases):
            self.single_test(test_case)

    def single_test(self, test_case):
        self.train_test_iter.df = self.df
        split_num = 1
        for train_df, test_df in self.train_test_iter:
            start_fit_time = timer()
            trainset = self.df2ds(train_df).build_full_trainset()
            test_case.alg.fit(trainset)
            end_fit_time = timer()
            fit_time = end_fit_time - start_fit_time
            start_pred_time = timer()
            preds = Tester.predict(test_case.alg, test_df)
            end_pred_time = timer()
            pred_time = end_pred_time - start_pred_time

            test_res_dict = dict(
                algorithm=test_case.alg_name,
                test_num=self.test_num,
                split_num=split_num,
                fit_time=fit_time,
                predict_time=pred_time,
                test_len=len(test_df),
                train_len=len(train_df),
            )

            test_res_dict["spearman"] = self.score_per_user(
                spearman, test_df, preds
            ).mean()

            test_res_dict["kendalltau"] = self.score_per_user(
                kendalltau, test_df, preds
            ).mean()

            test_res_dict["top_n_average"] = self.score_per_user(
                top_n_average_rating, test_df, preds
            ).mean()

            for score_fun_name, score_fun in self.score_functions.items():
                test_res_dict[score_fun_name] = score_fun(
                    preds["rating"], preds["prediction"]
                )

            for arg_name, arg in test_case.args.items():
                test_res_dict[arg_name] = arg

            single_test_res_df = pd.DataFrame(test_res_dict, index=[0])
            if self.first_test:
                results_df = single_test_res_df
                self.first_test = False
            else:
                results_df = pd.read_csv(test_case.filename)
                results_df = results_df.append(single_test_res_df, sort=False)
                results_df.reset_index(drop=True, inplace=True)

            results_df.to_csv(test_case.filename, index=False)
            split_num += 1
        self.vprint(".", end="")

    def score_per_user(self, scoring_fun, test_df, preds):
        results = np.array([])
        for userId in test_df["userId"].unique():
            userPreds = preds[preds["userId"] == userId]
            new_res = scoring_fun(
                userPreds["rating"].to_numpy(), userPreds["prediction"].to_numpy()
            )
            if np.isnan(new_res):
                results = np.append(results, 0.0)
            else:
                results = np.append(results, new_res)
        return results

    @staticmethod
    def predict(alg, df):
        df["prediction"] = df.apply(
            lambda row: alg.predict(row[0], row[1]).est, raw=True, axis=1,
        )
        return df[["userId", "rating", "prediction"]]


def movielens_df2ds(df):
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
