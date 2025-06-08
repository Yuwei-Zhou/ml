import random
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import talib
import vectorbt as vbt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from optuna.samplers import TPESampler

# 全局种子设置
np.random.seed(42)
random.seed(42)

# 配置参数
HOLDING_PERIOD = 2  # 持有到 t+HOLDING_PERIOD 日收盘（跨交易日持仓）
LOOKBACK_WINDOW = 20  # 回看窗口


# ==================== 加载数据 ====================
def load_data(filepath):
    """加载并预处理数据 - 修复日期处理版本"""
    try:
        data = pd.read_parquet(filepath)
        print(f"原始数据长度: {len(data)}")

        # 检查并处理索引
        if not isinstance(data.index, pd.DatetimeIndex):
            # 如果索引不是datetime，尝试转换
            if data.index.dtype == 'object':
                try:
                    data.index = pd.to_datetime(data.index)
                    print(f"成功转换索引为日期时间格式")
                except Exception as convert_error:
                    print(f"索引转换失败: {convert_error}")
                    # 如果有date列，使用date列作为索引
                    if 'date' in data.columns:
                        try:
                            data['date'] = pd.to_datetime(data['date'])
                            data = data.set_index('date')
                            print("使用date列作为索引")
                        except:
                            print("date列转换失败，保持原始索引")
                    else:
                        print("无法找到有效的日期列，保持原始索引")
            else:
                # 如果有date列，优先使用
                if 'date' in data.columns:
                    try:
                        data['date'] = pd.to_datetime(data['date'])
                        data = data.set_index('date')
                        print("使用date列作为索引")
                    except:
                        print("date列转换失败")

        # 排序确保时间顺序
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()

        # 数据基本信息
        if isinstance(data.index, pd.DatetimeIndex):
            print(f"数据时间范围: {data.index.min()} 到 {data.index.max()}")
        else:
            print(f"数据索引类型: {type(data.index)}")

        print(f"数据列: {list(data.columns)}")
        print(f"数据形状: {data.shape}")

        return data
    except Exception as e:
        print(f"数据加载错误: {e}")
        raise

def enhanced_features(data):
    """增强特征工程 - 合并版本，包含所有技术指标特征"""
    data = data.copy()

    # 基础特征
    data["returns"] = data["close"].pct_change().shift(1)
    data["volatility_5"] = data["returns"].rolling(5, min_periods=1).std().shift(1)
    data["volume_change"] = data["vol"].pct_change(3).shift(1)

    # 技术指标 - MACD
    try:
        data["macd"], data["macd_signal"], _ = talib.MACD(data["close"])
        data["macd"] = data["macd"].shift(1)
        data["macd_signal"] = data["macd_signal"].shift(1)
    except:
        data["macd"] = 0
        data["macd_signal"] = 0

    # OBV指标
    try:
        data["obv"] = talib.OBV(data["close"], data["vol"]).shift(1)
    except:
        data["obv"] = 0

    # 布林带特征 - 修复类型问题
    try:
        upper, mid, lower = talib.BBANDS(data["close"].values, timeperiod=20)
        upper_series = pd.Series(upper, index=data.index)
        lower_series = pd.Series(lower, index=data.index)
        mid_series = pd.Series(mid, index=data.index)

        data["bb_width"] = ((upper_series - lower_series) / mid_series).shift(1)
        data["bb_position"] = ((data["close"] - lower_series) /
                               (upper_series - lower_series)).shift(1)
    except:
        data["bb_width"] = 0
        data["bb_position"] = 0.5

    # RSI指标
    try:
        data["rsi"] = talib.RSI(data["close"], timeperiod=14).shift(1)
    except:
        data["rsi"] = 50

    # 随机指标
    try:
        data["stoch_k"], data["stoch_d"] = talib.STOCH(
            data["high"], data["low"], data["close"],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        data["stoch_k"] = data["stoch_k"].shift(1)
        data["stoch_d"] = data["stoch_d"].shift(1)
    except:
        data["stoch_k"] = 50
        data["stoch_d"] = 50

    # 价格特征
    data["range_pct"] = ((data["high"] - data["low"]) / data["close"]).shift(1)
    data["close_ma_ratio"] = (data["close"] / data["close"].rolling(50, min_periods=10).mean()).shift(1)

    # 量价背离指标
    price_change = data["close"].pct_change(5).shift(1)
    volume_change = data["vol"].pct_change(5).shift(1)
    data["pv_divergence"] = (price_change * volume_change < 0).astype(int)

    # 更多技术指标
    try:
        data["atr"] = talib.ATR(data["high"], data["low"], data["close"], timeperiod=14).shift(1)
        data["adx"] = talib.ADX(data["high"], data["low"], data["close"], timeperiod=14).shift(1)
        data["cci"] = talib.CCI(data["high"], data["low"], data["close"], timeperiod=20).shift(1)
    except:
        data["atr"] = (data["high"] - data["low"]).rolling(14).mean().shift(1)
        data["adx"] = 25
        data["cci"] = 0

    # 价格模式识别
    try:
        data["engulfing"] = talib.CDLENGULFING(data["open"], data["high"], data["low"], data["close"]).shift(1)
        data["hammer"] = talib.CDLHAMMER(data["open"], data["high"], data["low"], data["close"]).shift(1)
    except:
        data["engulfing"] = 0
        data["hammer"] = 0

    # Market regime features
    data['market_regime'] = np.where(
        data['close'] > data['close'].rolling(200, min_periods=50).mean(), 1, 0
    )

    # Multi-timeframe features
    for period in [5, 10, 20, 50]:
        data[f'momentum_{period}'] = data['close'].pct_change(period)
        data[f'vol_ratio_{period}'] = data['vol'] / data['vol'].rolling(period, min_periods=max(1, period // 4)).mean()

    # Price action patterns
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan).fillna(method='ffill').fillna(1)

    data['doji'] = np.abs(data['close'] - data['open']) / high_low_range
    data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / high_low_range
    data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / high_low_range

    # Volume-price relationship
    data['price_volume_trend'] = (data['close'].pct_change() * data['vol']).rolling(10, min_periods=1).sum()

    # VWAP calculation with error handling
    try:
        price_vol = (data['close'] * data['vol']).rolling(20, min_periods=1).sum()
        vol_sum = data['vol'].rolling(20, min_periods=1).sum().replace(0, np.nan)
        data['vwap'] = price_vol / vol_sum
        data['price_vs_vwap'] = data['close'] / data['vwap']
    except:
        data['vwap'] = data['close']
        data['price_vs_vwap'] = 1

    return data.fillna(method='ffill').fillna(0)

def dynamic_labeling(data):
    """动态标签生成 - 基于波动率调整持有期并生成改进的标签"""
    data = data.copy()

    # 确保有足够的数据
    if len(data) < HOLDING_PERIOD + 10:
        print(f"数据长度不足，需要至少 {HOLDING_PERIOD + 10} 条记录，当前只有 {len(data)} 条")
        data["target"] = np.nan
        return data.dropna(subset=["target"])

    # 动态持有期调整（基于波动率）
    if 'volatility_5' in data.columns:
        try:
            vol_series = data['volatility_5'].dropna()
            if len(vol_series) >= 10:
                # 分成三个波动率区间
                vol_low = vol_series.quantile(0.33)
                vol_high = vol_series.quantile(0.67)

                # 根据波动率调整持有期
                data['holding_period'] = np.where(
                    data['volatility_5'] <= vol_low,
                    HOLDING_PERIOD + 1,  # 低波动时延长持有期
                    np.where(
                        data['volatility_5'] >= vol_high,
                        max(1, HOLDING_PERIOD - 1),  # 高波动时缩短持有期
                        HOLDING_PERIOD  # 中等波动时使用默认持有期
                    )
                )
                print("使用动态持有期调整")
            else:
                data['holding_period'] = HOLDING_PERIOD
                print("波动率数据不足，使用固定持有期")
        except Exception as e:
            print(f"动态持有期调整失败: {e}")
            data['holding_period'] = HOLDING_PERIOD
    else:
        data['holding_period'] = HOLDING_PERIOD
        print("未找到volatility_5列，使用固定持有期")

    # 计算持有期收益（使用动态持有期）
    buy_price = data["open"].shift(-1)  # 次日开盘价买入

    # 对于动态持有期，需要逐行计算卖出价格
    sell_prices = []
    for i in range(len(data)):
        holding_days = int(data['holding_period'].iloc[i]) if pd.notna(
            data['holding_period'].iloc[i]) else HOLDING_PERIOD
        if i + holding_days < len(data):
            sell_prices.append(data["close"].iloc[i + holding_days])
        else:
            sell_prices.append(np.nan)

    sell_price = pd.Series(sell_prices, index=data.index)

    # 流动性过滤
    min_volume = data["vol"].rolling(20, min_periods=5).mean() * 0.3
    liquidity_mask = (data["vol"] > min_volume) & (data["vol"] > 0)

    # 计算未来收益率
    future_returns = np.where(
        liquidity_mask & (buy_price > 0) & (sell_price > 0),
        (sell_price / buy_price) - 1,
        np.nan
    )

    # 动态阈值设置
    valid_returns = future_returns[~np.isnan(future_returns)]
    if len(valid_returns) > 30:
        # 使用更合理的分位数
        # upper_threshold = max(np.percentile(valid_returns, 70), 0.01)
        # lower_threshold = min(np.percentile(valid_returns, 30), -0.01)
        returns_std = np.std(valid_returns)
        upper_threshold = np.mean(valid_returns) + 0.5 * returns_std  # e.g., mean + 0.5 standard deviations
        lower_threshold = np.mean(valid_returns) - 0.5 * returns_std  # e.g., mean - 0.5 standard deviations
    else:
        upper_threshold = 0.015
        lower_threshold = -0.015

    print(f"标签阈值: 买入>{upper_threshold:.3f}, 卖出<{lower_threshold:.3f}")

    # 三分类标签
    data["target"] = np.select(
        [
            future_returns > upper_threshold,
            (future_returns >= lower_threshold) & (future_returns <= upper_threshold),
            future_returns < lower_threshold
        ],
        [2, 1, 0],  # 2:买入, 1:持有, 0:卖出
        default=np.nan
    )

    # 移除无法计算标签的样本
    data.iloc[:1, data.columns.get_loc("target")] = np.nan

    # 动态移除末尾样本（基于最大持有期）
    max_holding_period = int(data['holding_period'].max()) if 'holding_period' in data.columns else HOLDING_PERIOD
    if max_holding_period > 0:
        data.iloc[-max_holding_period:, data.columns.get_loc("target")] = np.nan

    # 统计标签分布
    valid_labels = data["target"].dropna()
    print(f"标签统计:")
    print(f"总样本: {len(data)}")
    print(f"有效标签: {len(valid_labels)}")
    if len(valid_labels) > 0:
        print(valid_labels.value_counts().sort_index())
        if 'holding_period' in data.columns:
            print(f"持有期统计: 平均={data['holding_period'].mean():.1f}天, "
                  f"范围=[{data['holding_period'].min():.0f}, {data['holding_period'].max():.0f}]")

    return data.dropna(subset=["target"])

# ==================== 特征选择 ====================
def select_features(X_train, y_train):
    """时间序列特征选择"""
    estimator = RandomForestClassifier(
        n_estimators=100,
        n_jobs=1,
        random_state=42,
        max_features="sqrt",
    )
    selector = RFE(estimator, n_features_to_select=5)
    selector.fit(X_train, y_train)
    return selector

def safe_data_split(data, split_date_str="2024-01-01", min_train_ratio=0.7):
    """安全的数据分割函数 - 修复日期处理版本"""
    try:
        print(f"数据长度: {len(data)}")

        # 检查索引类型
        if isinstance(data.index, pd.DatetimeIndex):
            data_start = data.index.min()
            data_end = data.index.max()
            print(f"数据日期范围: {data_start} 到 {data_end}")

            # 检查数据日期范围是否合理
            if data_end <= data_start:
                raise ValueError("数据日期范围无效")

            # 计算数据跨度（天数）
            data_span_days = (data_end - data_start).days
            print(f"数据跨度: {data_span_days} 天")

            # 尝试使用指定日期分割
            try:
                split_date = pd.to_datetime(split_date_str)
                print(f"尝试使用分割日期: {split_date}")

                # 检查分割日期是否在数据范围内
                if split_date > data_start and split_date < data_end:
                    train_data = data[data.index < split_date]
                    test_data = data[data.index >= split_date]

                    # 验证分割结果
                    if len(train_data) >= 100 and len(test_data) >= 50:
                        print(f"日期分割成功:")
                        print(f"训练集: {len(train_data)} 样本 ({train_data.index.min()} 到 {train_data.index.max()})")
                        print(f"测试集: {len(test_data)} 样本 ({test_data.index.min()} 到 {test_data.index.max()})")
                        return train_data, test_data
                    else:
                        print(f"日期分割结果不理想 - 训练集:{len(train_data)}, 测试集:{len(test_data)}")
                else:
                    print(f"分割日期 {split_date} 不在数据范围内")
            except Exception as date_error:
                print(f"日期分割失败: {date_error}")
        else:
            print("索引不是日期时间格式，使用比例分割")

        # 使用比例分割作为后备方案
        split_idx = int(len(data) * min_train_ratio)
        split_idx = max(100, min(split_idx, len(data) - 50))  # 确保最小样本数

        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        print(f"使用比例分割 ({min_train_ratio:.1%}):")
        print(f"训练集: {len(train_data)} 样本")
        print(f"测试集: {len(test_data)} 样本")

        if isinstance(data.index, pd.DatetimeIndex):
            print(f"训练集时间范围: {train_data.index.min()} 到 {train_data.index.max()}")
            print(f"测试集时间范围: {test_data.index.min()} 到 {test_data.index.max()}")

        return train_data, test_data

    except Exception as e:
        print(f"数据分割错误: {e}")
        # 最后的保险措施
        split_idx = max(100, int(len(data) * 0.7))
        if split_idx >= len(data) - 10:
            split_idx = len(data) - 50
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        print(f"使用默认分割: 训练集{len(train_data)}样本, 测试集{len(test_data)}样本")
        return train_data, test_data

def train_model_ensemble(X, y, n_trials=100, use_stacking=True):
    """集成模型训练方法 - 包含堆叠集成功能的完整版本"""
    print(f"开始集成模型训练，样本数: {len(X)}, 特征数: {X.shape[1]}")

    # 检查数据有效性
    if len(X) < 50:
        raise ValueError(f"训练样本过少: {len(X)}, 需要至少50个样本")

    # 保存原始特征名称
    original_feature_names = list(X.columns)
    print(f"原始特征: {original_feature_names}")

    # 检查标签分布
    label_counts = pd.Series(y).value_counts()
    print(f"标签分布: {dict(label_counts)}")

    # 确保每个类别至少有足够样本
    min_class_samples = 5
    if any(label_counts < min_class_samples):
        print(f"警告: 某些类别样本过少，最少的类别只有 {label_counts.min()} 个样本")

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=original_feature_names, index=X.index)

    # 特征选择 - 使用RFE方法
    max_features = min(10, X_scaled.shape[1])
    selector = RFE(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        n_features_to_select=max_features
    )
    selector.fit(X_scaled, y)

    # 获取选中的特征
    selected_feature_mask = selector.support_
    selected_feature_names = [original_feature_names[i] for i, selected in enumerate(selected_feature_mask) if selected]
    print(f"选择的特征: {selected_feature_names}")

    # 应用特征选择
    X_selected = selector.transform(X_scaled)

    # 交叉验证策略 - 确保完全分区
    def get_cv_strategy(X_len):
        """获取合适的交叉验证策略"""
        from sklearn.model_selection import KFold, StratifiedKFold

        # 检查标签分布，决定使用哪种交叉验证
        try:
            # 尝试使用分层交叉验证（适合分类问题）
            n_splits = min(5, max(3, X_len // 30))  # 确保每折至少有30个样本

            # 检查每个类别的样本数是否足够进行分层
            min_class_count = pd.Series(y).value_counts().min()
            if min_class_count >= n_splits:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_name = "StratifiedKFold"
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_name = "KFold"
        except Exception as e:
            # 如果出错，使用简单的KFold
            n_splits = min(3, max(2, X_len // 50))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_name = "KFold (fallback)"

        # 验证交叉验证是否能正确分割数据
        try:
            splits = list(cv.split(X_selected, y))
            if len(splits) < 2:
                raise ValueError("分割数量太少")

            # 检查是否为完全分区
            all_indices = set()
            for train_idx, test_idx in splits:
                if len(set(train_idx) & set(test_idx)) > 0:
                    raise ValueError("训练集和测试集有重叠")
                all_indices.update(test_idx)

            if len(all_indices) != X_len:
                raise ValueError("不是完全分区")

            print(f"使用交叉验证策略: {cv_name}, 折数: {len(splits)}")
            return cv

        except Exception as e:
            print(f"交叉验证验证失败: {e}, 使用简单的3折KFold")
            return KFold(n_splits=3, shuffle=True, random_state=42)

    cv_strategy = get_cv_strategy(len(X_selected))

    # 如果使用堆叠集成
    if use_stacking:
        print("使用堆叠集成方法...")

        # 导入堆叠集成所需的模块
        from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression

        # 基础模型
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42))
        ]

        # 如果样本足够多，添加更多模型
        if len(X_selected) > 200:
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                base_models.append(('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)))
                print("添加梯度提升分类器到基础模型中")
            except ImportError:
                print("无法导入梯度提升分类器，跳过")
                pass

        # 添加LightGBM到基础模型（如果可用）
        try:
            import lightgbm as lgb
            from sklearn.base import BaseEstimator, ClassifierMixin

            # 创建LightGBM包装器
            class LGBMWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, num_leaves=31, learning_rate=0.1, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=5, min_child_samples=20,
                             num_boost_round=50, random_state=42):
                    self.num_leaves = num_leaves
                    self.learning_rate = learning_rate
                    self.feature_fraction = feature_fraction
                    self.bagging_fraction = bagging_fraction
                    self.bagging_freq = bagging_freq
                    self.min_child_samples = min_child_samples
                    self.num_boost_round = num_boost_round
                    self.random_state = random_state
                    self.model = None
                    self.classes_ = None
                    self._feature_importances = None

                def fit(self, X, y):
                    self.classes_ = np.unique(y)

                    # 计算类别权重
                    if len(self.classes_) > 1:
                        class_weights = compute_class_weight("balanced", classes=self.classes_, y=y)
                        weight_dict = dict(zip(self.classes_, class_weights))
                        weights = np.array([weight_dict[label] for label in y])
                    else:
                        weights = np.ones(len(y))

                    params = {
                        'objective': 'multiclass',
                        'num_class': len(self.classes_),
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': self.num_leaves,
                        'learning_rate': self.learning_rate,
                        'feature_fraction': self.feature_fraction,
                        'bagging_fraction': self.bagging_fraction,
                        'bagging_freq': self.bagging_freq,
                        'min_child_samples': self.min_child_samples,
                        'verbosity': -1,
                        'random_state': self.random_state,
                        'force_row_wise': True
                    }

                    train_data = lgb.Dataset(X, label=y, weight=weights)
                    self.model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=self.num_boost_round,
                        callbacks=[lgb.log_evaluation(period=0)]
                    )

                    # 保存特征重要性
                    if self.model is not None:
                        self._feature_importances = self.model.feature_importance()

                    return self

                def predict(self, X):
                    if self.model is None:
                        raise ValueError("模型尚未训练")

                    # 修复：LightGBM predict方法调用
                    y_pred = self.model.predict(X)
                    if y_pred.ndim > 1:
                        return np.argmax(y_pred, axis=1)
                    else:
                        return (y_pred > 0.5).astype(int)

                def predict_proba(self, X):
                    if self.model is None:
                        raise ValueError("模型尚未训练")

                    # 修复：LightGBM predict方法调用
                    y_pred = self.model.predict(X)
                    if y_pred.ndim == 1:
                        # 二分类情况
                        proba = np.column_stack([1 - y_pred, y_pred])
                    else:
                        # 多分类情况
                        proba = y_pred
                    return proba

                @property
                def feature_importances_(self):
                    """返回特征重要性"""
                    return self._feature_importances

            base_models.append(('lgb', LGBMWrapper()))
            print("添加LightGBM到基础模型中")
        except ImportError:
            print("LightGBM不可用，跳过")
            pass

        print(f"基础模型: {[name for name, _ in base_models]}")

        # 创建堆叠分类器 - 使用更保守的设置
        try:
            stacking_classifier = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(random_state=42, max_iter=1000),
                cv=cv_strategy,
                stack_method='predict_proba',
                n_jobs=1,  # 避免并行处理问题
                passthrough=False  # 不传递原始特征
            )
        except Exception as e:
            print(f"创建堆叠分类器时出错: {e}")
            # 如果还是有问题，使用最简单的设置
            from sklearn.model_selection import cross_val_score
            stacking_classifier = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(random_state=42, max_iter=1000),
                cv=3,  # 使用固定的3折
                stack_method='predict_proba',
                n_jobs=1
            )

        # 训练集成模型
        print("训练堆叠集成模型...")
        try:
            stacking_classifier.fit(X_selected, y)
            print("堆叠集成模型训练成功")
        except Exception as e:
            print(f"堆叠集成训练失败: {e}")
            print("回退到简单的投票集成...")

            # 回退方案：使用投票分类器
            from sklearn.ensemble import VotingClassifier
            voting_classifier = VotingClassifier(
                estimators=base_models,
                voting='soft'  # 使用软投票
            )
            voting_classifier.fit(X_selected, y)
            stacking_classifier = voting_classifier
            print("使用投票集成模型替代")

        final_model = stacking_classifier
        feature_importance = None

        # 尝试获取特征重要性（从基础模型中）
        try:
            importances = []
            weights = []  # 用于加权平均

            # 从训练好的堆叠模型中获取基础估计器
            if hasattr(final_model, 'estimators_'):
                estimators = final_model.estimators_
            else:
                # 如果是投票分类器，直接使用原始基础模型
                estimators = [model for name, model in base_models]

            for i, estimator in enumerate(estimators):
                try:
                    if hasattr(estimator, 'feature_importances_'):
                        importance = estimator.feature_importances_
                        if importance is not None and len(importance) == len(selected_feature_names):
                            importances.append(importance)
                            weights.append(1.0)  # 等权重
                            print(f"成功获取模型 {i} 的特征重要性")
                except Exception as e:
                    print(f"无法从模型 {i} 获取特征重要性: {e}")
                    continue

            if importances:
                # 计算加权平均特征重要性
                importances_array = np.array(importances)
                weights_array = np.array(weights)
                weights_array = weights_array / weights_array.sum()  # 归一化权重

                avg_importance = np.average(importances_array, axis=0, weights=weights_array)
                feature_importance = pd.DataFrame({
                    "feature": selected_feature_names,
                    "importance": avg_importance
                }).sort_values("importance", ascending=False)
                print(f"成功计算特征重要性，使用了 {len(importances)} 个模型")
            else:
                print("无任何模型提供特征重要性信息")

        except Exception as e:
            print(f"计算特征重要性时出错: {e}")

        print("堆叠集成模型训练完成")

    else:
        # 使用原始的单模型优化方法
        print("使用单模型优化方法...")

        # Optuna优化目标函数
        def objective(trial):
            # 选择模型类型
            model_type = trial.suggest_categorical('model_type', ['lgb', 'rf'])

            if model_type == 'lgb':
                # LightGBM参数
                params = {
                    'objective': 'multiclass',
                    'num_class': len(np.unique(y)),
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'verbosity': -1,
                    'random_state': 42,
                    'force_row_wise': True
                }
            else:  # RandomForest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42
                }

            scores = []

            try:
                for fold, (train_idx, valid_idx) in enumerate(cv_strategy.split(X_selected)):
                    X_train_fold = X_selected[train_idx]
                    X_valid_fold = X_selected[valid_idx]
                    y_train_fold = y.iloc[train_idx]
                    y_valid_fold = y.iloc[valid_idx]

                    if len(X_valid_fold) < 10:
                        continue

                    if model_type == 'lgb':
                        # 计算类别权重
                        classes = np.unique(y_train_fold)
                        if len(classes) > 1:
                            class_weights = compute_class_weight("balanced", classes=classes, y=y_train_fold)
                            weight_dict = dict(zip(classes, class_weights))
                            weights = np.array([weight_dict[label] for label in y_train_fold])
                        else:
                            weights = np.ones(len(y_train_fold))

                        # 创建Dataset
                        train_data = lgb.Dataset(X_train_fold, label=y_train_fold, weight=weights)
                        valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)

                        # 训练模型
                        model = lgb.train(
                            params,
                            train_data,
                            valid_sets=[valid_data],
                            num_boost_round=50,
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=10, verbose=False),
                                lgb.log_evaluation(period=0)
                            ]
                        )

                        # 预测和评估 - 修复：移除 num_iteration 参数
                        y_pred = model.predict(X_valid_fold)
                        if y_pred.ndim > 1:
                            y_pred_class = np.argmax(y_pred, axis=1)
                        else:
                            y_pred_class = (y_pred > 0.5).astype(int)

                    else:  # RandomForest
                        model = RandomForestClassifier(**params)
                        model.fit(X_train_fold, y_train_fold)
                        y_pred_class = model.predict(X_valid_fold)

                    score = np.mean(y_pred_class == y_valid_fold)
                    scores.append(score)

            except Exception as e:
                print(f"交叉验证错误: {e}")
                return 0.0

            if not scores:
                return 0.0

            return np.mean(scores)

        # 优化超参数
        try:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            print(f"最佳参数: {study.best_params}")
            print(f"最佳得分: {study.best_value}")

            best_params = study.best_params.copy()
            best_model_type = best_params.pop('model_type')

        except Exception as e:
            print(f"超参数优化失败，使用默认参数: {e}")
            best_model_type = 'lgb'
            best_params = {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20
            }

        print(f"使用最佳模型类型: {best_model_type}")

        # 训练最终模型
        if best_model_type == 'lgb':
            # 添加必要的LightGBM参数
            best_params.update({
                'objective': 'multiclass',
                'num_class': len(np.unique(y)),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'random_state': 42,
                'force_row_wise': True
            })

            # 计算类别权重
            classes = np.unique(y)
            if len(classes) > 1:
                class_weights = compute_class_weight("balanced", classes=classes, y=y)
                weight_dict = dict(zip(classes, class_weights))
                weights = np.array([weight_dict[label] for label in y])
            else:
                weights = np.ones(len(y))

            # 训练最终LightGBM模型
            train_data = lgb.Dataset(X_selected, label=y, weight=weights)
            final_model = lgb.train(
                best_params,
                train_data,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(period=0)]
            )

            # LightGBM特征重要性
            feature_importance = pd.DataFrame({
                "feature": selected_feature_names,
                "importance": final_model.feature_importance()
            }).sort_values("importance", ascending=False)

        else:  # RandomForest
            # 训练最终RandomForest模型
            final_model = RandomForestClassifier(**best_params)
            final_model.fit(X_selected, y)

            # RandomForest特征重要性
            feature_importance = pd.DataFrame({
                "feature": selected_feature_names,
                "importance": final_model.feature_importances_
            }).sort_values("importance", ascending=False)

    # 显示特征重要性
    if feature_importance is not None:
        print("\n特征重要性:")
        print(feature_importance)
    else:
        print("\n无法计算特征重要性")

    # 返回模型、特征选择器、缩放器和选中的特征名称
    return final_model, selector, scaler, selected_feature_names

def debug_signal_generation(data):
    """打印关键指标的统计信息，用于调试信号生成过程。"""
    print("\n信号生成关键指标统计：")
    if 'pred_proba' in data.columns:
        print(f"预测概率 > 0.5 的比例: {len(data[data['pred_proba'] > 0.5]) / len(data):.2%}")

    vol_condition = data['vol'] > data['vol'].rolling(20).mean()
    print(f"成交量过滤触发比例: {vol_condition.sum() / len(data):.2%}")

    ma_condition = data['close'] > data['close'].rolling(200).mean()
    print(f"处于200日均线上方比例: {ma_condition.sum() / len(data):.2%}")

# ==================== 交易信号生成（修复版） ====================

def improved_signal_generation(data, model, feature_selector, scaler, selected_features):
    """
    改进的信号生成逻辑 - 结合两个函数的最佳特性

    参数:
    - data: 输入数据DataFrame
    - model: 训练好的模型
    - feature_selector: 特征选择器
    - scaler: 数据标准化器
    - selected_features: 选择的特征列表
    """
    # 确保 data 是 DataFrame 对象的副本
    data = data.copy()

    print(f"信号生成 - 选择的特征: {selected_features}")

    # === 特征处理部分 (来自 generate_signals) ===
    # 获取所有需要的特征（训练时使用的完整特征集）
    if hasattr(scaler, 'feature_names_in_'):
        all_training_features = list(scaler.feature_names_in_)
    else:
        # 如果没有该属性，使用默认特征列表
        all_training_features = [col for col in data.columns if
                                 col not in ['target', 'open', 'high', 'low', 'close', 'vol', 'date']]

    print(f"训练时使用的完整特征集: {all_training_features}")

    # 检查特征可用性
    missing_features = [f for f in all_training_features if f not in data.columns]
    if missing_features:
        print(f"警告: 缺少以下特征: {missing_features}")
        # 使用可用特征的交集
        available_features = [f for f in all_training_features if f in data.columns]
        if len(available_features) == 0:
            raise ValueError("没有可用的特征进行预测")
        all_training_features = available_features
        print(f"使用可用特征: {all_training_features}")

    # === 模型预测部分 (增强版) ===
    try:
        # 特征预处理
        X = data[all_training_features].fillna(method='ffill').fillna(0)
        X_scaled = scaler.transform(X)
        X_selected = feature_selector.transform(X_scaled)

        # 模型预测
        if hasattr(model, 'best_iteration'):
            pred_proba = model.predict(X_selected, num_iteration=model.best_iteration)
        else:
            if hasattr(model, "predict_proba"):
                pred_proba = model.predict_proba(X_selected)
            else:
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(X_selected)
                else:
                    pred_proba = model.predict(X_selected)

        # 处理多分类预测结果
        if pred_proba.ndim > 1:
            data["pred_proba"] = pred_proba[:, 2]  # 买入类别的概率
            data["pred_class"] = np.argmax(pred_proba, axis=1)
        else:
            data["pred_proba"] = pred_proba
            data["pred_class"] = (pred_proba > 0.5).astype(int)

        data["pred_proba"] = np.clip(data["pred_proba"], 0, 1)

    except Exception as e:
        print(f"特征预处理或预测错误: {e}")
        # 使用最小特征集进行预测 (fallback策略)
        try:
            min_features = selected_features[:min(3, len(selected_features))]
            available_min_features = [f for f in min_features if f in data.columns]

            if not available_min_features:
                # 如果连基本特征都没有，使用数据中可用的数值特征
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                available_min_features = [col for col in numeric_cols if
                                          col not in ['target', 'open', 'high', 'low', 'close', 'vol']][:3]

            X = data[available_min_features].fillna(method='ffill').fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            X_selected = X_scaled

            if hasattr(model, 'best_iteration'):
                pred_proba = model.predict(X_selected, num_iteration=model.best_iteration)
            else:
                pred_proba = model.predict(X_selected)


            if pred_proba.ndim > 1:
                data["pred_proba"] = pred_proba[:, 2]
                data["pred_class"] = np.argmax(pred_proba, axis=1)
            else:
                data["pred_proba"] = pred_proba
                data["pred_class"] = (pred_proba > 0.5).astype(int)

            data["pred_proba"] = np.clip(data["pred_proba"], 0, 1)

        except Exception as e2:
            print(f"备用预测也失败: {e2}")
            # 生成保守的预测作为最后备选
            data["pred_proba"] = np.random.random(len(data)) * 0.3 + 0.35
            data["pred_class"] = 1

    print(f"预测概率范围: {data['pred_proba'].min():.3f} - {data['pred_proba'].max():.3f}")

    # === 多步骤信号确认 (来自 improved_signal_generation) ===

    # 1. 动态概率阈值 (来自 generate_signals)
    prob_median = data["pred_proba"].rolling(50, min_periods=10).median().fillna(data["pred_proba"].median())
    dynamic_threshold = np.maximum(prob_median, 0.40)  # 最低60%
    primary_signal = data['pred_proba'] > dynamic_threshold

    # 2. 成交量确认 (增强版)
    # vol_ma = data['vol'].rolling(10, min_periods=5).mean()
    # volume_confirm = data['vol'] > vol_ma * 1.0
    volume_ma = talib.MA(data['vol'].values.astype(float), timeperiod=10)
    volume_confirm = data['vol'] > volume_ma

    # 3. 趋势确认 (多重确认)
    trend_short = data['close'] > data['close'].rolling(5, min_periods=1).mean()
    trend_long = data['close'] > data['close'].rolling(20, min_periods=10).mean()
    trend_confirm = trend_short & trend_long

    # 4. 波动率过滤（避免高波动期）
    if 'volatility_5' in data.columns:
        vol_threshold = data['volatility_5'].rolling(50, min_periods=10).quantile(0.9)
        vol_filter = data['volatility_5'] < vol_threshold
    else:
        # 计算简单波动率作为替代
        returns = data['close'].pct_change().rolling(5).std()
        vol_threshold = returns.rolling(50, min_periods=10).quantile(0.8)
        vol_filter = returns < vol_threshold

    # 5. 市场状态过滤
    if 'market_regime' in data.columns:
        market_filter = data['market_regime'] == 1  # 只在上升趋势中交易
    else:
        # 使用简单的市场状态判断
        market_trend = data['close'].rolling(20, min_periods=10).mean()
        market_filter = data['close'] > market_trend

    # === 信号生成 ===
    # 主信号：组合所有条件
    buy_conditions = (
            primary_signal &
            volume_confirm &
            vol_filter.fillna(True)
            # trend_confirm &
            # market_filter.fillna(True)
    )
    print(f"primary_signal 命中: {primary_signal.sum()}")
    print(f"volume_confirm 命中: {volume_confirm.sum()}")
    print(f"trend_confirm 命中: {trend_confirm.sum()}")
    print(f"vol_filter 命中: {vol_filter.sum()}")
    print(f"market_filter 命中: {market_filter.sum()}")

    data['entry_signal'] = buy_conditions.astype(int)
    # data['exit_signal'] = 0

    # === 退出信号生成 (来自 generate_signals) ===
    # HOLDING_PERIOD = 5  # 默认持有期
    # entry_indices = data[data["entry_signal"] == 1].index

    # for entry_date in entry_indices:
    #     try:
    #         if isinstance(data.index, pd.DatetimeIndex):
    #             # 对于日期索引，找到HOLDING_PERIOD天后的日期
    #             entry_pos = data.index.get_loc(entry_date)
    #             exit_pos = min(entry_pos + HOLDING_PERIOD, len(data) - 1)
    #             exit_date = data.index[exit_pos]
    #         else:
    #             # 对于数值索引
    #             entry_pos = data.index.get_loc(entry_date)
    #             exit_pos = min(entry_pos + HOLDING_PERIOD, len(data) - 1)
    #             exit_date = data.index[exit_pos]
    #
    #         data.loc[exit_date, "exit_signal"] = 1
    #     except (KeyError, IndexError):
    #         continue
    data['exit_signal'] = False

    # === Kelly准则仓位管理 ===
    try:
        # 简化的Kelly计算（如果没有历史函数）
        recent_signals = data['entry_signal'].rolling(100, min_periods=20).sum()
        if len(data) > 50 and recent_signals.iloc[-1] > 5:
            # 基于最近信号表现估算Kelly比例
            win_rate = 0.6  # 假设胜率
            avg_win_loss_ratio = 1.5  # 假设盈亏比
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            data['position_size'] = np.clip(kelly_fraction, 0.1, 0.3)
        else:
            data['position_size'] = 0.2
    except Exception as e:
        print(f"Kelly计算失败: {e}")
        data['position_size'] = 0.2

    # === 信号统计输出 ===
    print(f"\n信号统计:")
    print(f"买入信号: {data['entry_signal'].sum()}")
    print(f"卖出信号: {data['exit_signal'].sum()}")
    print(f"信号密度: {data['entry_signal'].sum() / len(data) * 100:.2f}%")
    print(f"平均仓位大小: {data['position_size'].mean():.3f}")

    return data

# def calculate_historical_win_rate(data, lookback=100):
#     """计算历史胜率"""
#     if len(data) < lookback:
#         return 0.6  # 默认胜率
#
#     # 简化计算：基于过去的信号表现
#     recent_data = data.tail(lookback)
#     if 'returns' in recent_data.columns:
#         positive_returns = (recent_data['returns'] > 0).sum()
#         total_trades = len(recent_data[recent_data['entry_signal'] == 1])
#         if total_trades > 0:
#             return positive_returns / total_trades
#
#     return 0.6  # 默认值
#
# def calculate_win_loss_ratio(data, lookback=100):
#     """计算盈亏比"""
#     if len(data) < lookback or 'returns' not in data.columns:
#         return 1.5, 1.0  # 默认盈亏比
#
#     recent_data = data.tail(lookback)
#     winning_trades = recent_data[recent_data['returns'] > 0]['returns']
#     losing_trades = recent_data[recent_data['returns'] < 0]['returns']
#
#     if len(winning_trades) > 0 and len(losing_trades) > 0:
#         avg_win = winning_trades.mean()
#         avg_loss = abs(losing_trades.mean())
#         return avg_win, avg_loss
#
#     return 1.5, 1.0  # 默认值

# ==================== 风险控制 ====================
def advanced_risk_management(data):
    """高级风险管理和仓位控制"""
    data = data.copy()

    # 基础仓位设置
    base_position_size = 0.2

    # 1. 波动率阈值计算和基础仓位调整
    if 'volatility_5' in data.columns:
        data["vol_threshold"] = (
            data["volatility_5"]
            .shift(1)
            .rolling(20, min_periods=5)
            .median()
        )

        # 基于波动率的基础仓位
        data["position_size"] = np.where(
            data["volatility_5"] > data["vol_threshold"],
            0.5,  # 高波动时50%仓位
            0.8,  # 正常情况80%仓位
        )

        # 波动率缩放调整
        vol_median = data['volatility_5'].rolling(100, min_periods=20).median()
        vol_scale = data["volatility_5"] / data["volatility_5"].rolling(100).mean()
        vol_scale = vol_scale.fillna(1)

        # 综合波动率调整
        vol_adjusted_multiplier = 1 / np.maximum(vol_scale, 1)
        data["position_size"] = data["position_size"] * vol_adjusted_multiplier

        # 基于历史波动率中位数的进一步调整
        vol_adjustment = base_position_size / (
                data['volatility_5'] / vol_median.fillna(data['volatility_5'].median())
        )
        vol_adjustment = vol_adjustment.fillna(base_position_size)
        data["position_size"] = np.minimum(data["position_size"], vol_adjustment)
    else:
        data["position_size"] = base_position_size

    # 2. 基于近期表现的动态仓位调整
    if 'returns' in data.columns:
        recent_returns = data['returns'].rolling(20, min_periods=5).mean()
        performance_multiplier = 1 + np.clip(recent_returns, -0.5, 0.5)
        performance_multiplier = performance_multiplier.fillna(1)
        data['position_size'] *= performance_multiplier

    # 3. 最大回撤控制（综合两种方法）
    if len(data) > 20:
        # 短期回撤控制（20天）
        rolling_max_20 = data["close"].rolling(20, min_periods=1).max()
        daily_drawdown = (data["close"] - rolling_max_20) / rolling_max_20
        max_drawdown_20 = daily_drawdown.rolling(20).min().fillna(0)

        # 长期回撤控制（50天）
        rolling_max_50 = data['close'].rolling(50, min_periods=10).max()
        drawdown_50 = (data['close'] - rolling_max_50) / rolling_max_50

        # 综合回撤控制
        drawdown_multiplier = np.where(
            (max_drawdown_20 < -0.1) | (drawdown_50 < -0.1),  # 回撤超过10%
            0.5,  # 仓位减半
            1.0
        )
        data["position_size"] *= drawdown_multiplier

    # 4. 止损和止盈设置
    if 'atr' in data.columns:
        data['stop_loss'] = data['close'] - (data['atr'] * 2)
        data['take_profit'] = data['close'] + (data['atr'] * 3)
    else:
        data['stop_loss'] = data['close'] * 0.95  # 5%止损
        data['take_profit'] = data['close'] * 1.08  # 8%止盈

    # 5. 仓位限制设置
    max_single_position = 0.15  # 单笔最大15%
    max_total_exposure = 0.4  # 总暴露最大40%
    min_position = 0.05  # 最小仓位5%

    # 应用最大仓位限制
    data['position_size'] = np.minimum(data['position_size'], max_single_position)

    # 6. 最终仓位范围限制
    data["position_size"] = np.clip(data["position_size"], min_position, max_single_position)

    return data

def robust_feature_selection(X, y, max_features=10):
    """稳健的特征选择方法"""
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.model_selection import TimeSeriesSplit

    if len(X) < 50:
        # 数据太少时简单选择
        n_features = min(max_features, X.shape[1])
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance.head(n_features)['feature'].tolist()

    # 多种特征选择方法
    selectors = {}
    try:
        selectors['f_test'] = SelectKBest(f_classif, k=min(max_features, X.shape[1]))
        selectors['f_test'].fit(X, y)
    except:
        pass

    try:
        selectors['mutual_info'] = SelectKBest(mutual_info_classif, k=min(max_features, X.shape[1]))
        selectors['mutual_info'].fit(X, y)
    except:
        pass

    # 时间序列稳定性测试
    tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 50))
    feature_stability = {}

    try:
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < 20 or len(val_idx) < 10:
                continue

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]

            rf = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=5)
            rf.fit(X_train_fold, y_train_fold)

            for i, feature in enumerate(X.columns):
                importance = rf.feature_importances_[i]
                feature_stability.setdefault(feature, []).append(importance)
    except Exception as e:
        print(f"特征稳定性测试失败: {e}")

    # 选择稳定的特征
    stable_features = []
    for feature, importance_list in feature_stability.items():
        if len(importance_list) >= 2:
            stability = np.std(importance_list) / (np.mean(importance_list) + 1e-8)
            if stability < 1.0:  # 相对稳定
                stable_features.append((feature, np.mean(importance_list)))

    # 按重要性排序
    stable_features.sort(key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in stable_features[:max_features]]

    # 如果稳定特征不够，用RandomForest补充
    if len(selected_features) < max_features:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        additional_features = []
        for feature in feature_importance['feature']:
            if feature not in selected_features:
                additional_features.append(feature)
                if len(selected_features) + len(additional_features) >= max_features:
                    break

        selected_features.extend(additional_features)

    return selected_features[:max_features]

# ==================== 回测引擎 ====================
def backtest_strategy(data):
    """
    执行回测 - 兼容旧版 vectorbt (v0.27.3) 的版本.
    使用基于百分比的止损/止盈作为后备方案.
    """
    try:
        # 检查必要列是否存在
        required_cols = ["close", "entry_signal", "exit_signal"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

        # 检查信号数量
        if data["entry_signal"].sum() == 0:
            print("警告: 没有买入信号，回测无法执行。")
            # 返回一个空的投资组合对象以避免后续错误
            return vbt.Portfolio.from_signals(data['close'], entries=False, exits=False, freq='1D', init_cash=100000)

        # --- WORKAROUND FOR OLDER vectorbt VERSIONS ---
        # This is the required method for your environment.
        # It calculates a fixed percentage for SL/TP based on the data's average characteristics.
        if 'atr' in data.columns and 'close' in data.columns and data['close'].mean() > 0:
            avg_close = data['close'].mean()
            avg_atr = data['atr'].mean()

            # Your logic uses SL=2*ATR, TP=3*ATR. We convert this to a percentage.
            sl_percent = (avg_atr * 2) / avg_close
            tp_percent = (avg_atr * 3) / avg_close
            print(f"Info: Using calculated percentage-based stops. SL: {sl_percent:.2%}, TP: {tp_percent:.2%}")
        else:
            # If ATR isn't available, use hardcoded default percentages
            sl_percent = 0.05  # 5% stop-loss
            tp_percent = 0.08  # 8% take-profit
            print(f"Warning: Using hardcoded percentage-based stops. SL: {sl_percent:.2%}, TP: {tp_percent:.2%}")

        pf = vbt.Portfolio.from_signals(
            close=data["close"],
            entries=data["entry_signal"] == 1,
            exits=data["exit_signal"],
            sl_stop=sl_percent,  # Use sl_stop with a percentage (float)
            tp_stop=tp_percent,  # Use tp_stop with a percentage (float)
            freq="1D",
            init_cash=100000,
            fees=0.001,
            size=0.95,
        )
        return pf

    except Exception as e:
        print(f"回测错误: {e}")
        # Fallback to the simplest backtest if the above fails
        try:
            print("尝试最简化回测...")
            pf = vbt.Portfolio.from_signals(
                close=data["close"],
                entries=data["entry_signal"] == 1,
                freq="1D",
                init_cash=100000,
            )
            return pf
        except Exception as e2:
            print(f"简化回测也失败: {e2}")
            raise

# ==================== 结果分析 ====================
def plot_results(pf):
    """可视化策略绩效结果"""
    try:
        pf.plot(subplots=["orders", "trade_pnl", "cum_returns", "drawdowns"]).show()
    except Exception as e:
        print(f"绘图错误: {e}")
        # 只显示基础图表
        pf.plot().show()

def validate_features(data, required_features):
    """验证数据中是否包含所需特征"""
    missing_features = [f for f in required_features if f not in data.columns]
    if missing_features:
        print(f"缺少特征: {missing_features}")
        print(f"可用特征: {list(data.columns)}")
        return False
    return True

def create_feature_pipeline(feature_names):
    """创建特征处理管道"""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), feature_names)
        ]
    )

    return preprocessor


# ==================== 主程序 ====================
if __name__ == "__main__":
    try:
        # 数据准备 - 使用示例数据文件路径
        # 请根据实际情况修改文件路径
        data_file = "./data/300649.SZ.parquet"

        # 检查文件是否存在
        import os

        if not os.path.exists(data_file):
            print(f"数据文件不存在: {data_file}")
            print("请确认文件路径正确")
            exit(1)

        data = load_data(data_file)
        data = enhanced_features(data)
        data = dynamic_labeling(data)

        print(f"处理后数据长度: {len(data)}")

        # 按时间分割数据集
        train_data, test_data = safe_data_split(data, "2024-01-01", 0.7)

        if len(train_data) == 0 or len(test_data) == 0:
            print("训练集或测试集为空，请检查数据和分割日期")
            exit(1)

        # 检查标签分布
        print(f"训练数据: {len(train_data)}")
        print(train_data["target"].value_counts())
        print(f"测试数据: {len(test_data)}")
        print(test_data["target"].value_counts())

        # 选择特征
        features = [
            "volatility_5",
            "volume_change",
            "macd",
            "obv",
            "bb_width",
            "bb_position",
            # "pv_divergence",
            "rsi",
            "atr",
            "adx"
        ]

        # 确保所有特征都存在
        available_features = [f for f in features if f in train_data.columns]
        print(f"可用特征: {available_features}")

        if len(available_features) < 3:
            print("可用特征太少，无法进行训练")
            exit(1)

        # 模型训练
        print("开始训练模型...")
        model, feature_selector, scaler, selected_feature_names = train_model_ensemble(
            train_data[available_features],
            train_data["target"]
        )


        # 信号生成 - 修改这部分
        print("生成交易信号...")
        test_data_with_signals = improved_signal_generation(
            test_data,  # DataFrame 数据
            model,  # 训练好的模型
            feature_selector,  # 特征选择器
            scaler,  # 数据标准化器
            selected_feature_names  # 选中的特征名称列表
        )
        test_data_with_signals = advanced_risk_management(test_data_with_signals)

        print("信号触发调试：")
        debug_signal_generation(test_data_with_signals)

        # 回测执行
        print("执行回测...")
        pf = backtest_strategy(test_data_with_signals)

        # 结果分析
        print("\n策略绩效:")
        print(pf.stats())

        # 可视化
        print("生成图表...")
        plot_results(pf)

    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback

        traceback.print_exc()