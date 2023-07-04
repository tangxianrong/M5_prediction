# %%
import os
import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
import numpy as np
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from category_encoders import MEstimateEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from category_encoders import OneHotEncoder
from IPython import embed
# %%
sample_sub_df = pd.read_csv('..\input\sample_submission.csv', index_col=0)
embed()
# %%


class Dataset:
    def __init__(self, tgt_slide_d=28, train_period=28*5, feature_range=0):
        '''
        d1~d1913為訓練資料，共1913天，原始特徵提供。
        d1914~1941為驗證，共28天，應該是比賽結束後釋出。
        d1942~d1966為測試，共28天，隱藏起來的資料範圍。
        故訓練時應該用~D1885預測D1886~D1913
          驗證時應該用~D1913預測D1914~D1941
          測試時應該用~D1941預測D1942~D1966
          這樣驗證水準才會最近似於預測水準

        但若這樣會犧牲掉很多的資料，故若針對預測不同天的未來準備不同的資料會能使用到更近期的資料去訓練
        ex 預測下一天的資料可用~d1912預測~d1913
           預測下兩天的資料可用~d1911預測~d1913依此類推
        '''
        val_csv = '..\input\sales_train_validation.csv'
        eval_csv = '..\input\sales_train_evaluation.csv'
        sample_sub_csv = '..\input\sample_submission.csv'

        # val_df_id, val_df_d_sales = self.preprocess(val_csv) # 讀取資料
        # eval_df_id, eval_df_d_sales = self.preprocess(eval_csv) # 讀取資料

        val_data, val_id_org_col, val_id_onehot_col, val_d_sales_col = self.preprocess(
            val_csv)  # 讀取資料
        eval_data, eval_id_org_col, eval_id_onehot_col, eval_d_sales_col = self.preprocess(
            eval_csv)  # 讀取資料

        sample_sub_df = pd.read_csv(sample_sub_csv, index_col=0)

        # all_df_id = pd.concat([val_df_id,eval_df_id], axis=0)
        # all_df_d_sales = pd.concat([val_df_d_sales,eval_df_d_sales], axis=0)
        # sample_sub_df_id =all_df_id.loc(sample_sub_df.index)
        # sample_sub_df_sales = all_df_id.loc(sample_sub_df.index)

        # 確認sliding window的測試資料會有多少個
        tgt_number = train_period//tgt_slide_d
        # 計算出最早的tgt與最晚的tgt的時間差幾天
        real_train_period = (tgt_number-1)*tgt_slide_d
        # 計算出最多一筆資料能使用多少長度的特徵，資料總長度減去實際的training基期的資料範圍，再減去使用的模型loop。
        max_feature_usage = 1913 - real_train_period - 28
        if feature_range > max_feature_usage:
            Exception('feature_range>max_feature_usage')
        elif feature_range == 0:
            feature_range = max_feature_usage

        # self.df_train = val_df_d_sales.iloc[:,:-28]
        # self.df_val =  eval_df_d_sales.iloc[:,:-28]

        def get_data(tgt_d: int, df: pd.DataFrame, id_org_col: list, id_onehot_col: list, d_sales_col: list, is_eval=False):

            # print(id_onehot_col)
            if not is_eval:
                keys = df.index.to_list()
                values = [x+f'tgt_date_{tgt_d}' for x in df.index.to_list()]
                my_dict = dict(zip(keys, values))
                df = df.rename(index=my_dict)
                df_sales = df[d_sales_col]
                '''y'''
                if tgt_d == 0:
                    y = df_sales.iloc[:, -28:]
                else:
                    y = df_sales.iloc[:, -28-tgt_d:-tgt_d]
                y.columns = [f'F{i+1}' for i in range(y.shape[1])]
                '''X'''
                X = df_sales.iloc[:, -28-tgt_d-feature_range:-28-tgt_d]
                tgt_d = X.columns.to_list()[-1]

            else:
                '''y'''
                y = ''

                '''X'''
                df_sales = df[d_sales_col]
                X = df_sales.iloc[:, -feature_range:]
                tgt_d = X.columns.to_list()[-1]

            # 更改欄位名稱為前幾天

            X.columns = [f'Previous_Day_{i}' for i in range(X.shape[1], 0, -1)]
            # print(X.head())
            # print(df.head())
            # df.reindex(columns=id_onehot_col)
            X = pd.concat([X, df[id_onehot_col]], axis=1)
            return X, y

        def get_x(is_train=False):
            pass

        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        for tgt_d in range(0, train_period, tgt_slide_d):
            X_tmp, y_tmp = get_data(
                tgt_d, val_data, val_id_org_col, val_id_onehot_col, val_d_sales_col)

            self.X_train = pd.concat([self.X_train, X_tmp], axis=0)
            self.y_train = pd.concat([self.y_train, y_tmp], axis=0)

        self.X_val, self.y_val = get_data(
            0, eval_data, eval_id_org_col, eval_id_onehot_col, eval_d_sales_col)

        # self.X_train = val_df_d_sales.iloc[:,:-28]
        # self.X_train.columns=[f'D_{i}' for i in range(self.X_train.shape[1])]
        # self.y_train = val_df_d_sales.iloc[:,-28:]

        # self.X_val = eval_df_d_sales.iloc[:,28:-28]
        # self.X_val.columns=[f'D_{i}' for i in range(self.X_val.shape[1])]
        # self.y_val = eval_df_d_sales.iloc[:,-28:]
        self.X_val_test, _ = get_data(
            0, val_data, val_id_org_col, val_id_onehot_col, val_d_sales_col, is_eval=True)
        self.X_eval_test, _ = get_data(
            0, eval_data, eval_id_org_col, eval_id_onehot_col, eval_d_sales_col, is_eval=True)

        self.X_test = pd.concat([self.X_val_test, self.X_eval_test], axis=0)
        # self.X_test =X_all.reindex(sample_sub_df.index.to_list())

        # self.X_train = self.concat(self.X_train, val_df_id)
        # self.X_val = self.concat(self.X_val, eval_df_id)

        # all_id=pd.concat([val_df_id,eval_df_id],axis=0)
        # self.X_test = self.concat(self.X_test, all_id.loc[sample_sub_df.index.to_list()])

        # self.X_train, self.X_local_val, self.y_train, self.y_local_val = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # self.sales = pd.read_csv(submission_csv,index_col=0)
        # self._sales_processed = (
        #     sales
        #     .drop(["警示編號", "監控層級", "觸發說明"], axis=1)
        #     .assign(**sales.資料日期.dt.isocalendar(),
        #             month=sales.資料日期.dt.month,
        #             quarter=sales.資料日期.dt.quarter,
        #             hfy=sales.資料日期.dt.quarter.isin([3, 4])+1,
        #             bigMon=sales.資料日期.dt.month.isin([1, 3, 5, 7, 8, 10, 12]).astype(int))  # 將sales整理出更多的特徵資料
        # )
        # self.txn_c = txn[txn['tran_type'] == 'DEBIT'].drop("tran_type", axis=1)
        # self.txn_d = txn[txn['tran_type'] ==
        #                  'CREDIT'].drop("tran_type", axis=1)
        # self.txn_t = txn[txn['tran_type'] == 'TXN'].drop("tran_type", axis=1)

        # self._txn_processed = txn  # _get_data_from_sql整理出的txn資料
        # self._risk_processed = risk  # _get_data_from_sql整理出的risk資料
        # self._info_processed = info  # _get_data_from_sql整理出的info資料
        # self.ac = ac  # _get_data_from_sql從VW_NP_FSC_PARTY_ACCOUNT_BRIDGE整理出的客帳戶對照

        # self.train = self._get_data("train")  # VIP訓練資料
        # self.val = self._get_data("val")  # VIP驗證資料
        # self.test = self._get_data("test")  # VIP當天資料

        # self.seg_mapping = self._get_cust_seg_def()

    def preprocess(self, file_name):
        data = pd.read_csv(file_name, index_col=0)  # 讀取資料
        id_org_col = data.iloc[:, :5].columns.to_list()
        d_sales_col = data.iloc[:, 5:].columns.to_list()
        data_col_onehot = self._preprocess_obj_feature(data[id_org_col])
        id_onehot_col = data_col_onehot.columns.to_list()
        data = self.concat(data, data_col_onehot)
        return data, id_org_col, id_onehot_col, d_sales_col

    def concat(self, X_df, obj_df):

        return pd.concat([obj_df, X_df], axis=1)

    def _preprocess_obj_feature(self, obj_df):
        # onehot encoding feature
        self.encoder1 = OneHotEncoder()
        self.encoder1.fit(obj_df)
        obj_df = self.encoder1.transform(obj_df)

        return obj_df

    def preprocess_target(self, data, index):
        return data.loc[index].iloc[:, -28:]

# %%
# sample_sub_df=pd.read_csv('..\input\sample_submission.csv',index_col=0)
# sample_sub_df.index

# %%
# sample_sub_df.shape

# %%
# import pandas as pd

# # 创建示例 DataFrame
# data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
# df = pd.DataFrame(data, index=['row1', 'row2', 'row3'])


# %%
# df.index.to_list()

# %%

# # 更改索引标签
# new_index_names = {'row1': 'new_row1', 'row2': 'new_row2', 'row3': 'new_row3'}
# new_df = df.rename(index=new_index_names)

# print(new_df)


# %%
dataset = Dataset()


# %%
# dataset.y_train.columns[dataset.y_train.isna().any()].tolist()


# %%
# dataset.X_train.info()

# %%
# test=pd.read_csv('..\input\sales_train_evaluation.csv',index_col=0)

# %%
# test.loc['HOBBIES_1_001_CA_1_evaluation']

# %%
# test.head()

# %%
# train_data.X_train

# %%
class Models:
    def __init__(self, is_load=False, load_path='models'):
        '''


        '''

        package_dir = os.path.dirname(os.path.abspath(os.getcwd()))
        self.model_path = os.path.join(package_dir, load_path)
        os.makedirs(self.model_path, exist_ok=True)

        if is_load:
            self._load_models(load_path)
        else:
            self._set_28_models()

    # def _create_pipe(self, X_train):
    #     return make_pipeline(
    #         TargetEncoder(cols=X_train.columns[X_train.dtypes == 'object'],
    #                       handle_missing='return_nan'),
    #         SimpleImputer(missing_values=np.nan, strategy='mean'),
    #         XGBRegressor()
    #     )
    def _load_models(self, load_path):

        self.models = [joblib.load(os.path.join(self.model_path, fn))
                       for fn in sorted(os.listdir(self.model_path))]

    def _set_28_models(self):
        self.models = []
        for i in range(28):
            self.models.append(XGBRegressor())

    def _train_28_models(self, X_train, X_local_val, y_train, y_local_val):
        for i in range(28):

            # from IPython import embed
            # embed()
            self.models[i].fit(X_train, y_train.iloc[:, i])
            self.models[i]
            y_pred = self.models[i].predict(X_local_val)
            # 計算均方根誤差
            mse = mean_squared_error(y_local_val.iloc[:, i], y_pred)
            rmse = np.sqrt(mse)
            print("mse:", mse, "     Root Mean Squared Error:", rmse)
            joblib.dump(self.models[i], os.path.join(
                self.model_path, f"model_day{str(i).zfill(2)}.pkl"), compress=3)

    def models_predict(self, X_val: pd.DataFrame, name='result_simple.csv'):

        output = X_val.iloc[:, 0:1]
        for i in range(28):
            val_data_tmp_y = self.models[i].predict(X_val)
            output[f'F{i+1}'] = val_data_tmp_y

        output = output.iloc[:, 1:]
        output.to_csv(name)


# %%
models = Models()
# 519 17 5.429
# 526 18 4.6
# 7*10

# %%
models._train_28_models(dataset.X_train, dataset.X_val,
                        dataset.y_train, dataset.y_val)

# %%
# output = dataset.X_test.iloc[:,0:1]
# for i in range(28):
#     val_data_tmp_y = models.models[i].predict(dataset.X_test)
#     output[f'F{i+1}'] = val_data_tmp_y

# output=output.iloc[:,1:]

# %%
# output.to_csv('result_simple.csv')

# %%
# output = val_data.x.iloc[:,0:1]
# for i in range(28):
#     val_data_tmp_y = models.models[i].predict(val_data.x)
#     output[f'F{i+1}'] = val_data_tmp_y

# output=output.iloc[:,1:]

# %%
# the_result=pd.concat([val_data.sales.iloc[:,0], output], axis=1)

# %%
# the_result.to_csv('first_result.csv',index=False)

# %%
# models.models[0].predict(train_data.X_local_val)

# %%
# train_data.y_local_val.iloc[:,0]

# %%
# mse = mean_squared_error(y_local_val[:,i], y_pred)

# %%
# models.models[0].predict(train_data.X_local_val)

# %%
models = Models(is_load=True)

# %%
models.models_predict(dataset.X_test, name='result2.csv')

# %%
# train_data.y_train.iloc[:,0]


# %%
# df.info()

# %%
# import pandas as pd

# # 创建示例DataFrame
# data = {'A': [1, 2, 3], 'B': ['foo', 'bar', 'baz'], 'C': [True, False, True]}
# df = pd.DataFrame(data)

# # 获取所有物件类型的列
# object_columns = df.select_dtypes(exclude='object').columns

# # 打印列名
# print(df[object_columns])
# # 这将打印出DataFrame中所有物件类型的列名。你可以根据需要使用这些列名进行进一步的操作，例如筛选特定的列或进行其他数据处理。


# %%
# for x in train_data.sales.index:
#     print(x)

# %%


# class model:
#     def __init__(self, load=False):


#         package_dir = os.path.dirname(os.path.abspath(__file__))
#         self.model_path = os.path.join(
#             package_dir, f'./models/')

#         os.makedirs(self.model_path, exist_ok=True)

#         if load:
#             self.models = [joblib.load(os.path.join(self.model_path, fn))
#                            for fn in sorted(os.listdir(self.model_path))]
#         else:
#             for fn in os.listdir(self.model_path):
#                 os.remove(os.path.join(self.model_path, fn))
#             self.models = []
#             self._train_inference = None
#             self._train_27_models()

#     def _create_pipe(self, X_train):
#         return make_pipeline(
#             TargetEncoder(cols=X_train.columns[X_train.dtypes == 'object'],
#                           handle_missing='return_nan'),
#             SimpleImputer(missing_values=np.nan, strategy='mean'),
#             XGBRegressor()
#         )
#     def _train_27_models(self):
#         for i in range(1, 28):
#             train = self._create_label("train", i)
#             x_train, y_train = train.drop(
#                 self.cols_dropped, axis=1), train.警示數量
#             # from IPython import embed
#             # embed()
#             model = self._create_pipe(x_train)
#             model.fit(x_train, y_train)
#             joblib.dump(model, os.path.join(
#                 self.model_path, f"{datetime.now().date()}_w{str(i).zfill(2)}.pkl"), compress=3)
#             self.models.append(model)


# %%
# model.get_booster().get_score()
