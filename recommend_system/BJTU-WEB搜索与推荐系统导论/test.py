#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pandas as pd

df = pd.DataFrame({"user_id": [1, 2, 2, 1],
                   "env": [1, 2, 2, 1],
                   "env2": [2, 2, 3, 2],
                   "is_click": [0, 1, 0, 1]})

df_user = df.groupby(by=['user_id', 'env']).agg(
    u_env_count=pd.NamedAgg(column="is_click", aggfunc="sum"),
)
df_user = df_user.reset_index()
print(df_user)

column_name = 'u_env2_cnt'
df_user = df.groupby(by=['user_id', 'env2']).agg(
    count=pd.NamedAgg(column="is_click", aggfunc="sum"),
)
df_user.rename(columns={'count': column_name}, inplace=True)
df_user = df_user.reset_index()
print(df_user)
