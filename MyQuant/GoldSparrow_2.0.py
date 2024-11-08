from pathlib import Path
from typing import Union
from qlib.contrib.rolling.base import Rolling
# from qlib.tests.data import GetData
# import fire
# from qlib import auto_init

CONF_LIST = ["/home/godlike/project/GoldSparrow/workflow_config_lstm_Alpha158.yaml"]

class RollingBenchmark(Rolling):
    """_summary_

    Args:
        Rolling (_type_): _description_
    """

    def __init__(self,
                 conf_path: Union[str, Path] = None,
                 horizon=40,
                 **kwargs) -> None:

        print('conf_path', conf_path)
        super().__init__(conf_path=conf_path, horizon=horizon, **kwargs)




if __name__ == "__main__":
    #####################################
    # 0 删除缓存数据集handler pkl文件
    #####################################
    import os
    from os import listdir
    pkl_path = os.path.dirname(__file__)  # 当前文件所在的目录
    for file_name in listdir(pkl_path):
        if file_name.endswith('.pkl'):
            os.remove(pkl_path + '/' + file_name)

    ###################################
    # 1 滚动训练与预测
    ###################################
    import qlib
    exp_name = "combine"  # 合并预测结果pred.pkl存放mlflow实验名
    rb = RollingBenchmark(
        conf_path=CONF_LIST[0],
        step=40,  # 滚动步长，每隔40天滚动训练一次，它也决定了每滚测试集长度为20天
        horizon=5,  # 收益率预测期长度
        exp_name=exp_name)  # 最终合并预测结果所在实验名

    config_dict = rb._raw_conf()  # 配置字典
    # 初始化qlib
    qlib.init(provider_uri=config_dict["qlib_init"]["provider_uri"],
              region=config_dict["qlib_init"]["region"])

    # 滚动训练与预测
    rb._train_rolling_tasks()

    #################################
    # 2 滚动预测结果合并成大预测结果：每步小测试期预测结果合并成大测试期预测结果
    #################################
    rb._ens_rolling()
    # 打印合并后预测结果文件所在实验id，实验名和记录id
    print('experiment_id', rb._experiment_id, 'exp_name', exp_name, 'rid',
          rb._rid)

    #################################
    # 3 qlib信号分析与回测：在大测试期里执行信号分析与回测
    #################################
    # 回测:记录信号分析结果（如IC等）和回测结果（如仓位情况等）
    rb._update_rolling_rec()

    # 打印合并后预测结果文件所在实验id，实验名和记录id。回测结果也在此实验和记录id下。
    print('experiment_id', rb._experiment_id, 'exp_name', exp_name, 'rid',
          rb._rid)
