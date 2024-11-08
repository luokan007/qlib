from pprint import pprint

import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.workflow import R
import quantstats as qs

if __name__ == "__main__":

    qlib.init(provider_uri=r"G:\qlibrolling\qlib_data\cn_data")
    exp_name = "combine"
    rid = "cb93ce30d9f54c08842b3becf8cefb2e"
    predict_recorder = R.get_recorder(recorder_id=rid,
                                      experiment_name=exp_name)
    pred_df_all = predict_recorder.load_object('pred.pkl')  # 加载预测文件
    start_date = pred_df_all.index[0][0]
    end_date = pred_df_all.index[-1][0]
    CSI300_BENCH = "SH000300"
    FREQ = "day"
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred_df_all,
    }

    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
        "verbose": False, # 是否日志输出订单信息
    }

    backtest_config = {
        "start_time": start_date,
        "end_time": end_date,
        "account": 100000000,
        "benchmark": CSI300_BENCH,
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "open",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)

    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj,
                                                     strategy=strategy_obj,
                                                     **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))

    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"], freq=analysis_freq)
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"] -
        report_normal["cost"],
        freq=analysis_freq)
    analysis["return_with_cost"] = risk_analysis(report_normal["return"] -
                                                 report_normal["cost"],
                                                 freq=analysis_freq)
    analysis["return_bench"] = risk_analysis(report_normal["bench"],
                                             freq=analysis_freq)
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    # log metrics
    analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
    # print out results
    pprint(
        f"The following are analysis results of benchmark return({analysis_freq})."
    )
    pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
    pprint(
        f"The following are analysis results of the excess return without cost({analysis_freq})."
    )
    pprint(analysis["excess_return_without_cost"])
    pprint(
        f"The following are analysis results of the excess return with cost({analysis_freq})."
    )
    pprint(analysis["excess_return_with_cost"])
    pprint(
        f"The following are analysis results of the return with cost({analysis_freq})."
    )
    pprint(analysis["return_with_cost"])
    pprint(
        f"The following are analysis results of the return of benchmark({analysis_freq})."
    )
    pprint(analysis["return_bench"])

    output = "quantstats-tearsheet_qlib.html"
    returns = report_normal["return"] - report_normal["cost"]
    qs.reports.html(returns, benchmark=report_normal["bench"], output=output)

    import webbrowser
    webbrowser.open(output)
