#!/usr/bin/env python3
# -*- coding:utf-8 -*--
CAT_FEATURES = [
    'ctx_net_env',
    'ctx_time_expose_hour_bucket',
    'ctx_time_expose_hour',
    'ctx_time_expose_is_opening_hours',
    'ctx_time_expose_day_of_week',
    'ctx_time_expose_is_weekend',

    'u_device',
    'u_os',
    'u_province',
    'u_city',
    'u_age_max',
    'u_sex_max',

    'i_category1',
    'i_category2',
]


SELECT_COLUMNS = [
    'user_id',
    'u_device',
    'u_os',
    'u_province',
    'u_city',
    'u_age_max',
    'u_sex_max',
    'u_total_cnt',
    'u_click_cnt',
    'u_ctr',

    'u_click_ctx_diff_days_mean',
    'u_click_ctx_diff_days_std',
    'u_click_ctx_time_expose_hour_mean',
    'u_click_ctx_time_expose_hour_std',
    'u_click_i_keys_len_mean',
    'u_click_i_keys_len_std',

    'item_id',
    # 'i_click_cnt',
    'i_cnt_img',
    'i_category1',
    'i_category2',
    'i_title_len',
    'i_keys_len',

    'ctx_net_env',
    'u_cnt_flush',
    'ctx_timestamp_expose',
    # 'ctx_time_expose_hour_bucket',
    'ctx_time_expose_hour',
    # 'ctx_time_expose_is_opening_hours',
    'ctx_time_expose_day_of_week',
    # 'ctx_time_expose_is_weekend'
    'ctx_cross_i_time_release_diff_days',
    'ctx_cross_i_time_release_timestamp_diff',
    ################ AUC: 0.695 ################
]
