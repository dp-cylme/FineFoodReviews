#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pandas
import scipy

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

        self.__users = {}
        self.__current_user_num = 0

        self.__items = {}
        self.__current_item_num = 0

    def get_user(self, user):
        if user not in self.__users:
            self.__users[user] = self.__current_user_num
            self.__current_user_num += 1

        return self.__users[user]

    def get_item(self, item):
        if item not in self.__items:
            self.__items[item] = self.__current_item_num
            self.__current_item_num += 1

        return self.__items[item]

    def __df_to_sparsematrix(self, df:
                             pandas.DataFrame) -> scipy.sparse.coo.coo_matrix:
        sparsematrix = scipy.sparse.dok_matrix(
            (df[self.__user].value_counts().shape[0],
             df[self.__item].value_counts().shape[0]),
            dtype=numpy.int32)

        for _, row in df[[self.__user, self.__item, self.__rating]].iterrows():
            sparsematrix[self.get_user(row[0]), self.get_item(row[1])] = row[2]

        return sparsematrix.tocoo(copy=True)

    def fit(self, df: pandas.DataFrame):
        self.__model.fit(self.__df_to_sparsematrix(df), epochs=20)

    def predict(self,
                users: numpy.array,
                items: numpy.array,
                num_threads: int=1) -> numpy.array:
        return self.__model.predict(
            numpy.array([self.get_user(x) for x in users]),
            numpy.array([self.get_item(x) for x in items]),
            num_threads=num_threads)
