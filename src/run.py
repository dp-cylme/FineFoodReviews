#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd

from recsys import RecSys


def main():
    # Preprocessing

    # Processing (Здесь добавляем фичи)
    train_df = (pd.read_csv("../input/train.csv"))
    test_df = (pd.read_csv("../input/test.csv"))

    # Train (Здесь обчуаем модели и делаем из них ансамбль)
    recsys_model = RecSys("UserId", "ProductId", "Prediction")
    recsys_model.fit(train_df)

    # Prediction
    recsys_model.predict(test_df.UserId.values, test_df.ProductId.values)


if __name__ == "__main__":
    main()
