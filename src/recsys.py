#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pandas
import functools
import scipy

from typing import Any
from lightfm import LightFM


class RecSys(object):
    """
    Create rec sys model with lightfm
    """

    def __init__(self,
                 user_column_name: str="user",
                 item_column_name: str="item",
                 rating_column_name: str="rating"):
        self.__user = user_column_name
        self.__item = item_column_name
        self.__rating = rating_column_name

        self.__model = LightFM(learning_rate=0.05, loss='bpr')

    def __df_to_sparsematrix(self, df: pandas.DataFrame):
        def add_smth(smth_dict, smth_num, smth):
            if smth not in smth_dict:
                smth_dict[smth] = smth_num
                smth_num += 1
            return smth_dict[smth]

        self.__users = {}
        current_user_num = 0
        get_user = functools.partial(add_smth, self.__users, current_user_num)

        self.__items = {}
        current_item_num = 0
        get_item = functools.partial(add_smth, self.__items, current_item_num)

        sparsematrix = scipy.sparse.dok_matrix((df[self.__user].value_counts(
        ).shape[0], df[self.__item].value_counts().shape[0]))

        for _, row in df[[self.__user, self.__item, self.__rating]].iterrows():
            sparsematrix[get_user(row[0])][get_item(row[1])] = row[2]

        return sparsematrix

    def fit(self, df: pandas.DataFrame) -> Any:
        self.__model.fit(self.__df_to_sparsematrix(df), epochs=10)

    def predict(self, users: numpy.array, items: numpy.array) -> numpy.array:
        return self.__model.predict(users, items, num_threads=8)
