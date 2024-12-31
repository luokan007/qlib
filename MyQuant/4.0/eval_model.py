import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple


logger = logging.getLogger(__name__)

class MyEval:
    """模型评估类"""

    def __init__(self, pred=None, label=None, label_col: int = 0):
        """初始化评估器"""
        self.pred = self._preprocess_data(pred)
        self.label = self._preprocess_data(label)
        self.label_col = label_col

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据，统一索引格式"""
        if df is None:
            return None

        if not isinstance(df.index, pd.MultiIndex):
            logger.warning("Input DataFrame does not have MultiIndex")
            return df

        # 确保索引名称正确
        df.index.names = ['datetime', 'instrument']
        return df

    @classmethod
    def from_pkl(cls, pred_path: str, label_path: str, **kwargs):
        """从pkl文件初始化"""
        try:
            pred_df = pd.read_pickle(pred_path)
            label_df = pd.read_pickle(label_path)
            return cls(pred=pred_df, label=label_df, **kwargs)
        except Exception as e:
            logger.error(f"Error loading pkl files: {str(e)}")
            raise

    @classmethod
    def from_dataframe(cls, pred_df: pd.DataFrame, label_df: pd.DataFrame, **kwargs):
        """从DataFrame初始化"""
        return cls(pred=pred_df, label=label_df, **kwargs)
    
    def eval(self) -> Dict:
        """评估模型性能"""
        if self.label is None or not isinstance(self.label, pd.DataFrame) or self.label.empty:
            logger.warning("Empty label.")
            return {}

        ic, ric = self._calc_ic(self.pred.iloc[:, 0],
                                self.label.iloc[:, 0])
        # long_pre, short_pre = self._calc_long_short_prec(self.pred.iloc[:, 0], 
        #                                                  self.label.iloc[:, 0], 
        #                                                  is_alpha=True)


        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
            # "Long precision": long_pre.mean(),
            # "Short precision": short_pre.mean(),
        }

        return metrics

    @staticmethod
    def _calc_ic(pred: pd.Series, label: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """计算IC值"""
        pd_concat = pd.concat([pred, label], axis=1, keys=["pred", "label"])
        
        # 确保使用正确的索引级别名称
        if isinstance(pd_concat.index, pd.MultiIndex):
            ic = pd_concat.groupby(level=0).apply(lambda x: x["pred"].corr(x["label"]))
            ric = pd_concat.groupby(level=0).apply(lambda x: x["pred"].corr(x["label"], method="spearman"))
        else:
            logger.warning("Input data does not have MultiIndex structure")
            ic = pd.Series([pd_concat["pred"].corr(pd_concat["label"])])
            ric = pd.Series([pd_concat["pred"].corr(pd_concat["label"], method="spearman")])
        
        return ic, ric
        
    @staticmethod
    def _calc_long_short_prec(pred: pd.Series, 
                             label: pd.Series, 
                             quantile: float = 0.2,
                             is_alpha: bool = False) -> Tuple[pd.Series, pd.Series]:
        """计算多空准确率"""
        if is_alpha:
            label = label - label.mean(level="datetime")
        if int(1 / quantile) >= len(label.index.get_level_values(1).unique()):
            raise ValueError("Need more instruments to calculate precision")

        df = pd.DataFrame({"pred": pred, "label": label})
        group = df.groupby(level="datetime")

        def N(x):
            return int(len(x) * quantile)

        long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label).reset_index(level=0, drop=True)
        short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label).reset_index(level=0, drop=True)

        groupll = long.groupby("datetime")
        l_dom = groupll.apply(lambda x: x > 0)
        l_c = groupll.count()

        groups = short.groupby("datetime")
        s_dom = groups.apply(lambda x: x < 0)
        s_c = groups.count()
        return (l_dom.groupby("datetime").sum() / l_c), (s_dom.groupby("datetime").sum() / s_c)
    
if __name__ == "__main__":
    test_pred_path = "/home/godlike/project/qlib/qlib/mlruns/2/e43b0f4014c44fe5860cc39717d9d1d6/artifacts/pred.pkl"
    test_label_path = "/home/godlike/project/qlib/qlib/mlruns/2/e43b0f4014c44fe5860cc39717d9d1d6/artifacts/label.pkl"
    
    
    eval = MyEval.from_pkl(test_pred_path, test_label_path)
    print(eval.eval())