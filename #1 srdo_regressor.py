#1 srdo_regressor
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from kedro_work.utils import get_joblib_memory

memory = get_joblib_memory()

# https://arxiv.org/abs/1911.12580

class SrdoRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator=None, epsilon=1e-7):
        self.estimator = estimator
        self.epsilon = epsilon

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        memory.reduce_size()
        w = calc_decorr_weight(X, self.epsilon)
        fitted = clone(self.estimator)
        fitted.fit(X, y, sample_weight=w)
        self.estimator_ = fitted

        return self

    def predict(self, X):
        return self.estimator_.predict(X)

@memory.cache
def calc_decorr_weight(X, epsilon):
    classifier = lgb.LGBMClassifier(n_jobs=-1, random_state=0)

    X_positive = []
    for i in range(X.shape[1]):
        X_positive.append(np.random.choice(X[:, i], size=X.shape[0], replace=True))
    X_positive = np.array(X_positive).transpose()

    classifier.fit(
        np.vstack([X, X_positive]),
        np.concatenate([np.zeros(X.shape[0]), np.ones(X.shape[0])])
    )
    proba = classifier.predict_proba(X)
    w = proba[:, 1] / (epsilon + proba[:, 0])

    return w / np.sum(w)

#2 ridge_feature_count_scaler

from sklearn.base import BaseEstimator, TransformerMixin

# 特徴量をコピーしてN倍にしたときに、Ridgeのalphaを変えなくて良いようにスケーリング
class RidgeFeatureCountScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X)

        return X / (X.shape[1] ** 0.5)

    def inverse_transform(self, X, y=None):
        X = self._validate_data(X)

        return X * (X.shape[1] ** 0.5)

#3 positive_homogeneous_regressor       

import numpy as np
from sklearn.base import BaseEstimator, clone
from .utils import my_fit

# Positive Homogeneous
# https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_4Rin120/_pdf/-char/ja

class PositiveHomogeneousRegressor(BaseEstimator):
    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.regressor_ = clone(self.regressor)

        X_norm = np.sum(X.values ** 2, axis=1) ** 0.5
        X = X / (1e-37 + X_norm).reshape(-1, 1)
        y = y / (1e-37 + X_norm)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        sample_weight *= X_norm ** 2

        if fit_context is not None:
            fit_context = fit_context.copy()
            X_norm = np.sum(fit_context['X_val'].values ** 2, axis=1) ** 0.5
            fit_context['X_val'] = fit_context['X_val'] / (1e-37 + X_norm).reshape(-1, 1)
            fit_context['y_val'] = fit_context['y_val'] / (1e-37 + X_norm)
            if fit_context['sample_weight_val'] is None:
                fit_context['sample_weight_val'] = np.ones(fit_context['X_val'].shape[0])
            fit_context['sample_weight_val'] *= X_norm ** 2

        my_fit(
            self.regressor_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        return self

    def predict(self, X):
        X_norm = np.sum(X.values ** 2, axis=1) ** 0.5
        X = X / (1e-37 + X_norm).reshape(-1, 1)

        y = self.regressor_.predict(X)
        return y * (1e-37 + X_norm)

#4 parquet_dataset

from kedro.io.core import (
    AbstractDataSet
)

import pandas as pd

class ParquetDataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self):
        return pd.read_parquet(self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath)

    def _save(self, data) -> None:
        pass

#5 optuna_bbc_cv

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

# https://arxiv.org/abs/1708.07180

class OptunaBbcCv(BaseEstimator):
    def __init__(self, create_model=None, sampler=None, n_trials=None, cv=None, scoring_y_pred=None, features=None):
        self.create_model = create_model
        self.sampler = sampler
        self.n_trials = n_trials
        self.cv = cv
        self.scoring_y_pred = scoring_y_pred
        self.features = list(features)

    def fit(self, X=None, y=None, sample_weight=None):
        cv = list(self.cv.split(X))

        y_preds = []
        def objective(trial):
            model = self.create_model(trial)

            if sample_weight is not None:
                y_pred = np.zeros(X.shape[0])
                X_filtered = self._filter_X(X)
                for train_idx, val_idx in cv:
                    model.fit(X_filtered.iloc[train_idx], y.iloc[train_idx], sample_weight=sample_weight.iloc[train_idx])
                    y_pred[val_idx] = model.predict(X_filtered.iloc[val_idx])
            else:
                y_pred = cross_val_predict(model, self._filter_X(X), y, cv=cv)

            score = self.scoring_y_pred(X, y, y_pred)
            y_preds.append(y_pred)
            return -score

        study = optuna.create_study(sampler=self._create_sampler())
        study.optimize(objective, n_trials=self.n_trials)

        model = self.create_model(study.best_trial)
        model.fit(self._filter_X(X), y, sample_weight=sample_weight)

        y_pred_oos = np.zeros(X.shape[0])
        for train_idx, val_idx in cv:
            scores = []
            for y_pred in y_preds:
                score = self.scoring_y_pred(X.iloc[train_idx], y.iloc[train_idx], y_pred[train_idx])
                scores.append(score)
            scores = np.array(scores)

            n_bests = 1
            selected_y_preds = []
            for trial_idx in np.argsort(scores)[-n_bests:]:
                selected_y_preds.append(y_preds[trial_idx][val_idx])

            y_pred_oos[val_idx] = np.mean(selected_y_preds, axis=0)

        self.study_ = study
        self.model_ = model
        self.y_preds_ = np.array(y_preds)
        self.y_pred_oos_ = y_pred_oos

        return self

    def predict(self, X=None):
        return self.model_.predict(self._filter_X(X))

    def _filter_X(self, X):
        if self.features is not None:
            return X[self.features]
        return X

    def _create_sampler(self):
        optuna_seed = 1
        if self.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=optuna_seed)
        elif self.sampler == 'tpe_mv':
            sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=optuna_seed)
        elif self.sampler == 'random':
            sampler = optuna.samplers.RandomSampler(seed=optuna_seed)
        return sampler

#6 numerai_dataset2

from kedro.io.core import (
    AbstractDataSet
)

import pandas as pd
import numerapi

class NumeraiDataset2(AbstractDataSet):
    def __init__(self, is_train):
        self._is_train = is_train
        self._napi = numerapi.NumerAPI(verbosity="info")

    def _load(self):
        url = self._get_dataset_url()
        df = pd.read_parquet(url)
        return df

    def _get_current_round(self):
        return self._napi.get_current_round(tournament=8)

    def _get_dataset_url(self):
        round = self._get_current_round()

        filename = 'numerai_training_data.parquet' if self._is_train else 'numerai_tournament_data.parquet'

        query = """
            query ($filename: String!) {
                dataset(filename: $filename)
            }
            """
        params = {
            'filename': filename
        }
        if round:
            query = """
                        query ($filename: String!, $round: Int) {
                            dataset(filename: $filename, round: $round)
                        }
                        """
            params['round'] = round
        return self._napi.raw_query(query, params)['data']['dataset']

    def _describe(self):
        return dict(is_train=self._is_train)

    def _save(self, data) -> None:
        pass

#7 numerai_dataset

from kedro.io.core import (
    AbstractDataSet
)

import tempfile
import pandas as pd
import numerapi
import requests, zipfile
from kedro_work.utils import get_joblib_memory

memory = get_joblib_memory()

@memory.cache
def download_url(url):
    r = requests.get(url)
    return r.content

class NumeraiDataset(AbstractDataSet):
    def __init__(self, is_train):
        self._is_train = is_train
        self._napi = numerapi.NumerAPI(verbosity="info")

    def _load(self):
        url = self._napi.get_dataset_url()

        with tempfile.TemporaryDirectory() as dir:
            cache_path = '{}/numerai_cache.zip'.format(dir)
            with open(cache_path, 'wb') as f:
                f.write(download_url(url))
            z = zipfile.ZipFile(cache_path)

            if self._is_train:
                fname = 'numerai_training_data.csv'
            else:
                fname = 'numerai_tournament_data.csv'

            df = pd.read_csv(z.open(fname), index_col=0)
        return df

    def _describe(self):
        return dict(is_train=self._is_train)

    def _save(self, data) -> None:
        pass
#8 nonstationary_feature_remover

from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import numpy as np
import pandas as pd

class NonstationaryFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, remove_count=None, remove_ratio=None):
        if remove_count and remove_ratio:
            raise Exception('remove_count and remove_ratio cannot be set simultaneously')
        self.remove_count = remove_count
        self.remove_ratio = remove_ratio

    def fit(self, X, y=None):
        X = self._validate_data(X)

        model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)

        model.fit(X, np.arange(X.shape[0]))
        importances = model.feature_importances_

        if self.remove_count:
            remove_count = self.remove_count
        else:
            remove_count = int(self.remove_ratio * X.shape[1])

        features = list(range(X.shape[1]))
        feature_imp = pd.DataFrame(zip(importances, features), columns=['value', 'feature'])
        feature_imp = feature_imp.sort_values('value')

        for i in range(X.shape[1] - remove_count, X.shape[1]):
            features.remove(int(feature_imp['feature'].iloc[i]))

        self.selected_features_ = np.array(features)

        return self

    def transform(self, X, y=None):
        X = self._validate_data(X)

        return X[:, self.selected_features_].copy()

    def inverse_transform(self, X, y=None):
        raise Exception('inverse_transform not implemented')

        
#9 my_keras_regressor2


import numpy as np
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import marshal
import types
import traceback

# 追加機能
# mc dropout
# early stopping (fit_context版)

class MyKerasRegressor2(KerasRegressor):
    def __init__(self, mc_count=None, fit_params={}, **kwargs):
        KerasRegressor.__init__(self, **kwargs)
        self.mc_count_ = mc_count
        self.fit_params_ = fit_params

    def fit(self, X, y, fit_context=None, **kwargs):
        if fit_context is None:
            return KerasRegressor.fit(self, X, y, **self.fit_params_, **kwargs)
        else:
            return KerasRegressor.fit(
                self,
                X,
                y,
                validation_data=(fit_context['X_val'], fit_context['y_val']),
                **self.fit_params_,
                **kwargs,
            )

    def predict(self, X=None):
        if self.mc_count_ is None or self.mc_count_ == 1:
            return KerasRegressor.predict(self, X)

        ys = []

        X = tf.data.Dataset.from_tensor_slices(X)
        X = X.batch(65536)

        for i in range(self.mc_count_):
            ys.append(KerasRegressor.predict(self, X))

        return np.mean(ys, axis=0)

    def get_params(self, **params):
        res = KerasRegressor.get_params(self, **params)
        res.update({
            'mc_count': self.mc_count_,
            'fit_params': self.fit_params_,
        })
        return res

    def set_params(self, **params):
        self.mc_count_ = params['mc_count']
        self.fit_params_ = params['fit_params']
        params = params.copy()
        del params['mc_count']
        del params['fit_params_']
        return KerasRegressor.set_params(self, **params)

    # https://stackoverflow.com/questions/8574742/how-to-pickle-an-object-of-a-class-b-having-many-variables-that-inherits-from
    def __getstate__(self):
        a_state = KerasRegressor.__getstate__(self)
        b_state = {
            'mc_count_': self.mc_count_,
        }
        return (a_state, b_state)

    def __setstate__(self, state):
        a_state, b_state = state
        self.mc_count_ = b_state['mc_count_']
        KerasRegressor.__setstate__(self, a_state)

#10 my_keras_regressor

import numpy as np
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import marshal
import types
import traceback

# 追加機能
# mc dropout
# early stopping

class MyKerasRegressor(KerasRegressor):
    def __init__(self, mc_count=None, split=None, fit_params={}, **kwargs):
        KerasRegressor.__init__(self, **kwargs)
        self.mc_count_ = mc_count
        self.split_ = split
        self.fit_params_ = fit_params

    def fit(self, X, y, **kwargs):
        if self.split_ is None:
            return KerasRegressor.fit(self, X, y, **self.fit_params_, **kwargs)
        else:
            train_idx, val_idx = self.split_(X)
            return KerasRegressor.fit(
                self,
                X[train_idx],
                y[train_idx],
                validation_data=(X[val_idx], y[val_idx]),
                **self.fit_params_,
                **kwargs,
            )

    def predict(self, X=None):
        if self.mc_count_ is None or self.mc_count_ == 1:
            return KerasRegressor.predict(self, X)

        ys = []

        X = tf.data.Dataset.from_tensor_slices(X)
        X = X.batch(65536)

        for i in range(self.mc_count_):
            ys.append(KerasRegressor.predict(self, X))

        return np.mean(ys, axis=0)

    def get_params(self, **params):
        res = KerasRegressor.get_params(self, **params)
        res.update({
            'mc_count': self.mc_count_,
            'split': self.split_,
            'fit_params': self.fit_params_,
        })
        return res

    def set_params(self, **params):
        self.mc_count_ = params['mc_count']
        self.split_ = params['split']
        self.fit_params_ = params['fit_params']
        params = params.copy()
        del params['mc_count']
        del params['split']
        del params['fit_params_']
        return KerasRegressor.set_params(self, **params)

    # https://stackoverflow.com/questions/8574742/how-to-pickle-an-object-of-a-class-b-having-many-variables-that-inherits-from
    def __getstate__(self):
        a_state = KerasRegressor.__getstate__(self)
        b_state = {
            'mc_count_': self.mc_count_,
            # 'split_': marshal.dumps(self.split_.__code__),
            # 'split_': self.split_,
        }
        return (a_state, b_state)

    def __setstate__(self, state):
        a_state, b_state = state
        self.mc_count_ = b_state['mc_count_']
        # code = marshal.loads(b_state['split_'])
        # self.split_ = types.FunctionType(code, globals(), "some_func_name")
        # self.split_ = b_state['split_']
        KerasRegressor.__setstate__(self, a_state)



#11 my_fit

import inspect
import lightgbm as lgb
import xgboost as xgb

def my_fit(model, *args, **kwargs):
    if kwargs.get('fit_context') is not None:
        fit_context = kwargs['fit_context']
        if isinstance(model, lgb.LGBMRegressor) or isinstance(model, lgb.LGBMClassifier):
            kwargs['eval_set'] = [(fit_context['X_val'], fit_context['y_val'])]
            if 'sample_weight_val' in fit_context and fit_context['sample_weight_val'] is not None:
                kwargs['eval_sample_weight'] = [fit_context['sample_weight_val']]
            kwargs['early_stopping_rounds'] = fit_context['early_stopping_rounds']
            kwargs['verbose'] = False
            del kwargs['fit_context']
            print('early stopping is used lgbm')

        if isinstance(model, xgb.XGBRegressor) or isinstance(model, xgb.XGBClassifier):
            kwargs['eval_set'] = [(fit_context['X_val'], fit_context['y_val'])]
            if 'sample_weight_val' in fit_context and fit_context['sample_weight_val'] is not None:
                kwargs['eval_sample_weight'] = [fit_context['sample_weight_val']]
            kwargs['early_stopping_rounds'] = fit_context['early_stopping_rounds']
            kwargs['verbose'] = False
            del kwargs['fit_context']
            print('early stopping is used xgb')

    argspec = inspect.getfullargspec(model.fit)
    # print(argspec)
    if 'fit_context' in kwargs and 'fit_context' not in argspec.args:
        del kwargs['fit_context']

    # print(model)
    # print(kwargs.keys())
    # print(argspec.args)
    # print(argspec)
    #
    # if 'sample_weight' in kwargs and 'sample_weight' not in argspec.args:
    #     del kwargs['sample_weight']

    return model.fit(*args, **kwargs)

    #12 mlflow_utils
import mlflow
import yaml
import matplotlib.pyplot as plt
import cloudpickle
import tempfile
import lzma

class MlflowPlot():
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        return None

    def __exit__(self, type, value, traceback):
        with tempfile.TemporaryDirectory() as dir:
            fname = '{}/{}'.format(dir, self.filename)
            plt.savefig(fname, bbox_inches='tight') # tightでlegendが収まるようになる
            plt.close('all')
            mlflow.log_artifact(fname)

def mlflow_plot(filename):
    return MlflowPlot(filename)

def mlflow_log_model(model, path):
    if not path.endswith('.xz'):
        raise Exception('mlflow_log_model path must end with .xz')

    data = cloudpickle.dumps(model)
    data = lzma.compress(data)
    with tempfile.TemporaryDirectory() as dir:
        fname = '{}/{}'.format(dir, path)
        with open(fname, 'wb') as f:
            f.write(data)
        mlflow.log_artifact(fname)

def mlflow_log_yaml(obj, path):
    with tempfile.TemporaryDirectory() as dir:
        fname = '{}/{}'.format(dir, path)
        with open(fname, "w") as f:
            yaml.dump(obj, f)
        mlflow.log_artifact(fname)

def mlflow_log_str(x, path):
    with tempfile.TemporaryDirectory() as dir:
        fname = '{}/{}'.format(dir, path)
        with open(fname, "w") as f:
            f.write(str(x))
        mlflow.log_artifact(fname)


#13 mlflow_artifact_dataset


from kedro.io.core import (
    AbstractDataSet
)

import joblib
from mlflow.tracking import MlflowClient
import tempfile

class MlflowArtifactDataset(AbstractDataSet):
    def __init__(self, run_id, artifact_path):
        self._run_id = run_id
        self._artifact_path = artifact_path

    def _load(self):
        with tempfile.TemporaryDirectory() as dest_path:
            client = MlflowClient()
            path = client.download_artifacts(
                run_id=self._run_id,
                path=self._artifact_path,
                dst_path=dest_path
            )
            return joblib.load(path)

    def _describe(self):
        return dict(run_id=self._run_id, artifact_path=self._artifact_path)

    def _save(self, data) -> None:
        pass

 #14   fear_greedy

 import pandas as pd
import requests
import json

def fetch_fear_greedy():
    url = 'https://api.alternative.me/fng/?limit=3000'
    df = pd.DataFrame(json.loads(requests.get(url).text)['data'])
    df = df[df['time_until_update'].isna()]
    df = df.drop(columns=['time_until_update', 'value_classification'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['value'] = df['value'].astype('float')
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    df = df.rename(columns={ 'value': 'fear_greedy_index' })
    return df

#15 era_boost_xgb_estimators
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score
from .utils import my_fit

class EraBoostXgbRegressor(BaseEstimator):
    def __init__(self, base_estimator=None, num_iterations=3, proportion=0.5, n_estimators=None):
        self.base_estimator = base_estimator
        self.num_iterations = num_iterations
        self.proportion = proportion
        self.n_estimators = n_estimators

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.base_estimator_ = clone(self.base_estimator)

        my_fit(
            self.base_estimator_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        for iter in range(self.num_iterations - 1):
            y_pred = self.base_estimator_.predict(X)

            era_scores = []
            indicies = []
            n = y_pred.shape[0]
            m = 10
            for i in range(m):
                idx = np.arange(i * n // m, (i + 1) * n // m)
                indicies.append(idx)
                y_pred2 = indexing(y_pred, idx)
                y2 = indexing(y, idx)
                era_scores.append(r2_score(y2, y_pred2))

            score_threshold = np.quantile(era_scores, self.proportion)
            idx = []
            for i in range(m):
                if era_scores[i] <= score_threshold:
                    idx.append(indicies[i])
            idx = np.concatenate(idx)

            self.base_estimator_.n_estimators += self.n_estimators
            booster = self.base_estimator_.get_booster()
            self.base_estimator_.fit(indexing(X, idx), indexing(y, idx), xgb_model=booster)

        return self

    def predict(self, X):
        return self.base_estimator_.predict(X)

def indexing(x, idx):
    if hasattr(x, 'iloc'):
        return x.iloc[idx]
    else:
        return x[idx]

#16 early_stopping_estimators

import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state
from sklearn.ensemble._base import _set_random_states
from .utils import my_fit

# https://proceedings.neurips.cc/paper/1996/file/f47330643ae134ca204bf6b2481fec47-Paper.pdf
ENSEMBLE_MODE_BALANCING = 'balancing'

class BaseEarlyStoppingEstimator(BaseEstimator):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 cv=None,
                 # max_samples=1.0,
                 # max_features=1.0,
                 ensemble_mode=None,
                 random_state=None,
                 verbose=0):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.cv = cv
        # self.max_samples = max_samples
        # self.max_features = max_features
        self.ensemble_mode = ensemble_mode
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        # n = X.shape[0]
        random_state = check_random_state(self.random_state)
        # count = round(self.max_samples * n)
        # feature_count = round(self.max_features * X.shape[1])

        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.estimators_features_ = []
        if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
            self.val_errors_ = []

        cv_gen = self.cv.split(X)

        for i in range(self.n_estimators):
            train_idx, val_idx = cv_gen.__next__()

            estimator = clone(self.base_estimator)
            _set_random_states(estimator, random_state=random_state.randint(np.iinfo(np.int32).max))

            sw = None if sample_weight is None else sample_weight[train_idx]

            fit_context = {
                'X_val': indexing(X, val_idx),
                'y_val': indexing(y, val_idx),
                'sample_weight_val': None if sample_weight is None else indexing(sample_weight, val_idx),
                'early_stopping_rounds': 100,
            }

            my_fit(
                estimator,
                indexing(X, train_idx),
                indexing(y, train_idx),
                sample_weight=sw,
                fit_context=fit_context,
            )

            if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
                y_val_pred = estimator.predict(X_val)
                val_error = np.average((y_val - y_val_pred) ** 2, weights=sw_val)
                self.val_errors_.append(val_error)

            # indicies = calc_indicies(n, count, random_state)
            # feature_indicies = calc_feature_indicies(X.shape[1], feature_count, random_state)

            feature_indicies = np.arange(X.shape[1])

            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_indicies)

        if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
            self.val_errors_ = np.array(self.val_errors_)

        return self

class EarlyStoppingRegressor(BaseEarlyStoppingEstimator):
    def predict(self, X):
        ys = []
        for i, estimator in enumerate(self.estimators_):
            ys.append(estimator.predict(indexing2(X, self.estimators_features_[i])))
        ys = np.array(ys)

        if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
            w = cp.Variable((len(self.estimators_), X.shape[0]))

            # 2 * w[i] * val_errors[i]
            # - w[i] * y[i] ** 2
            # + w[i] * w[j] * y[i] * y[j] -> sum(w[i] * y[i]) ** 2

            objective = cp.Minimize(
                2 * cp.sum(cp.multiply(w, np.repeat(self.val_errors_.reshape(-1, 1), X.shape[0], axis=1)))
                - cp.sum(cp.multiply(w, ys ** 2))
                + cp.sum(cp.multiply(w, ys)) ** 2
            )

            constraints = [
                0 <= w,
                cp.sum(w, axis=0) == 1,
            ]

            prob = cp.Problem(objective, constraints)
            try:
                result = prob.solve()
            except cp.error.SolverError:
                print('cvxpy solve failed. use equal weight')
                return np.mean(ys, axis=0)

            return np.sum(ys * w.value, axis=0)
        else:
            return np.mean(ys, axis=0)

class EarlyStoppingClassifier(BaseEarlyStoppingEstimator):
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        class_to_idx = {}
        for i, cls in enumerate(self.classes_):
            class_to_idx[cls] = i
        proba = np.zeros(X.shape[0], self.n_classes_)

        for estimator in self.estimators_:
            if hasattr(estimator, "predict_proba"):
                p = estimator.predict_proba(X)
                for i, cls in enumerate(estimator.classes_):
                    proba[:, class_to_idx[cls]] += p[:, i]
            else:
                y_pred = estimator.predict(X)
                for i, cls in enumerate(self.classes_):
                    proba[y_pred == cls, i] += 1

        return proba / self.n_estimators

def calc_indicies(n, count, random_state):
    indicies = random_state.randint(n, size=count)
    return np.sort(indicies)

def calc_feature_indicies(n, count, random_state):
    if n == count:
        return np.arange(n)
    else:
        return random_state.choice(np.arange(n), size=count, replace=False)

def indexing(x, idx):
    if hasattr(x, 'iloc'):
        return x.iloc[idx]
    else:
        return x[idx]

def indexing2(x, idx):
    if hasattr(x, 'iloc'):
        return x.iloc[:, idx]
    else:
        return x[:, idx]

#17 cv

import numpy as np

def _purge_idx(train_idx, val_idx, groups, purge):
    unique_groups = np.unique(groups[val_idx])
    purged_groups = unique_groups.reshape(1, -1) + np.arange(-purge, purge + 1).reshape(-1, 1)
    purged_groups = np.unique(purged_groups)
    return train_idx[~np.isin(groups[train_idx], purged_groups)]

def my_group_kfold(groups, n_splits=5, purge=12):
    if hasattr(groups, 'values'):
        groups = groups.values
    idx = np.arange(groups.size)
    g = np.sort(np.unique(groups))
    cv = []
    for i in range(n_splits):
        selected = g[i * g.size // n_splits:(i + 1) * g.size // n_splits]
        val_idx = np.isin(groups, selected)
        cv.append((
            _purge_idx(idx[~val_idx], idx[val_idx], groups, purge),
            idx[val_idx],
        ))
    return cv

def my_kfold(x, n_splits=5, purge=12):
    return my_group_kfold(np.arange(x.shape[0]), n_splits=n_splits, purge=purge)


#18 clf_sign_regressor

import numpy as np
from sklearn.base import BaseEstimator, clone
from .utils import my_fit

class ClfSignRegressor(BaseEstimator):
    def __init__(self, classifier=None):
        self.classifier = classifier

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.classifier_ = clone(self.classifier)

        sw = np.abs(y)
        if sample_weight is not None:
            sw *= sample_weight
        y = np.sign(y).astype('int')

        if fit_context is not None:
            fit_context = fit_context.copy()
            sw_val = np.abs(fit_context['y_val'])
            if fit_context['sample_weight_val'] is not None:
                sw_val *= fit_context['sample_weight_val']
            fit_context['y_val'] = np.sign(fit_context['y_val']).astype('int')
            fit_context['sample_weight_val'] = sw_val

        my_fit(
            self.classifier_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        return self

    def predict(self, X):
        proba = self.classifier_.predict_proba(X)
        return np.sum(proba * np.array(self.classifier_.classes_), axis=1)

#19 clf_binning_regressor

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import KBinsDiscretizer
from .utils import my_fit

class ClfBinningRegressor(BaseEstimator):
    def __init__(self, classifier=None, n_bins=None):
        self.classifier = classifier
        self.n_bins = n_bins

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.classifier_ = clone(self.classifier)
        self.transformer_ = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')

        y = self.transformer_.fit_transform(y.reshape(-1, 1)).flatten().astype('int')

        if fit_context is not None:
            fit_context = fit_context.copy()
            fit_context['y_val'] = self.transformer_.transform(fit_context['y_val'].reshape(-1, 1)).flatten().astype('int')

        my_fit(
            self.classifier_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        self.class_values_ = self.transformer_.inverse_transform(np.array(self.classifier_.classes_).reshape(-1, 1)).flatten()

        return self

    def predict(self, X):
        proba = self.classifier_.predict_proba(X)
        return np.sum(proba * self.class_values_, axis=1)
