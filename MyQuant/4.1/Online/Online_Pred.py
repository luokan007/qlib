import os
import json
import pickle
from pathlib import Path
from generate_order import GenerateOrder
import qlib
from qlib.data import D
from qlib.contrib.data.handler import MyAlpha158_DyFeature
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.processor import RobustZScoreNorm, DropnaProcessor, Fillna,  DropnaLabel, CSRankNorm, FilterCol

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def prepare_data(config):
    """_summary_

    Args:
        config (_type_): _description_

    Returns:
        _type_: _description_
    """

    cal = D.calendar(freq="day")
    latest_date = cal[-1]

    ##设置测试数据的时间范围
    test_start_time = latest_date
    test_end_time = latest_date

    pool = config['pool']
    start_time = config['train'][0]
    end_time = test_end_time
    fit_start_time = config['train'][0]
    fit_end_time = config['train'][1]
    feature_meta_file = config['feature_meta_file']

    selected_features = ["VWAP0","DAILY_AMOUNT_RATIO","TRIX_48","RESI10","ULTOSC","RESI5","STOCHF_k","QTLD5","KLOW2","STD30",
                        "TURNOVER","CORD30","ADOSC","MIN60","STD20","PBMQR","RSQR5","ADX_14","KUP2","RESI20",
                        "RSQR10","STD60","QTLU5","CORD20","STR_FACTOR","STOCHRSI_k","STOCHF_d","AD","RSV5","CCI_14",
                        "CORR5","WVMA5","WILLR_6","STD10","MA5","RESI30","IMXD60","KSFT","QTLD10","NATR_28",
                        "VSTD5","RANK60","STD5","LOW0","KMID","MA20","VSTD20","RSV60","CORR60","KSFT2",
                        "RESI60","ROC30","HIGH0","ROC5","CORD60","MFI_24","BETA5","CORD5","MIN30","VMA60",
                        "BETA20","VMA5","STOCHRSI_d","OPEN0","ADX_28","WILLR_48","OBV","KMID2","RSV10","CORD10",
                        "CORR10","TRANGE","KLEN","BOP","MFI_6","RSV30","BETA10","ROC_6","RSI_6","SUMD5",
                        "WILLR_12","MACD_HIST","RSI_12","CORR30","CCI_28","IMXD30","KLOW","PSTTM","norm_RSRS","MA60",
                        "MAX5","PETTM","KUP","RSQR30","QTLU10","VMA10","RSQR60","ROC10","ROC_12","MACD_SIGNAL",
                        "MA30","RANK30","RSV20","ROC60","VSTD60","MA10","BETA60","NATR_14","SUMN5","TRIX_24",
                        "WILLR_24","BETA30","RSQR20","MIN20","MAX60","QTLD20","MAX30","pos_RSRS","APO","CORR20",
                        "MIN10","MFI_48","WVMA60","MFI_12","MOM_6","TRIX_12","RANK10","QTLD60","ATR_14","WVMA10",
                        "ROC_24","QTLD30","AROON_14_down","MIN5","VMA20","SUMP5","VSUMP5","CMO_14","VSTD30","CNTN60",
                        "QTLU20","VSTD10","AROON_28_down","MAX20","RANK20","RANK5","MOM_12","RSI_24","CNTN5","ROC20",
                        "IMXD20","IMAX5","MOM_48","AROON_28_up","CMO_28","CNTD10","QTLU60","MAX10","VMA30","VSUMN5",
                        "IMIN60","IMXD5","WVMA20","SUMN10","CNTN30","IMXD10","IMIN10","ROC_48","SUMP10","MOM_24",
                        "IMIN30","TRIMA_12","TRIMA_48","CNTD30","SUMN60","IMAX60","TEMA_24","WVMA30","CNTN10","VSUMP60",
                        "VSUMP20","MACD","VSUMP10","TRIMA_24","TEMA_48","TEMA_12","SUMD10","SUMP60","QTLU30","CNTD20",
                        "SUMP30","KAMA_48","CNTN20","VSUMN10","ATR_28","SAR","KAMA_12","SUMN30","CNTP30","EMA_5",
                        "CNTP5","VSUMP30","SUMP20","IMIN5","IMAX10","CNTP10","VSUMN30","CNTD5","CNTD60","CNTP60",
                        "SUMD30","CNTP20","VSUMN60","IMAX20","VSUMN20","IMAX30","SUMN20","EMA_10","VSUMD5","SUMD60",
                        "IMIN20","VSUMD60","EMA_20","KAMA_24","VSUMD10","VSUMD20", "SUMD20","AROON_14_up","VSUMD30"]


    infer_processors = [FilterCol(fields_group='feature',
                                          col_list=selected_features),
                                RobustZScoreNorm(fit_start_time=fit_start_time,
                                             fit_end_time=fit_end_time,
                                            fields_group='feature',
                                            clip_outlier=True),
                                Fillna(fields_group='feature')]
    learn_processors = [DropnaLabel(), CSRankNorm(fields_group='label')]
    filter_rule = None

    handler =MyAlpha158_DyFeature(instruments=pool,
            start_time=start_time,
            end_time=end_time,
            freq="day",
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            filter_pipe=filter_rule,
            feature_meta_file=feature_meta_file)

    step_len = config['model_step_len']
    dataset = TSDatasetH(step_len = step_len, handler = handler, segments={'test': (test_start_time, test_end_time)})

    return dataset

def main():
    ## 生成订单文件
    basic_info_path = "/home/godlike/project/GoldSparrow/Meta_Data/stock_basic.csv"

    working_dir = "/home/godlike/project/GoldSparrow/"
    config_file_name = 'config_20250209121635.json'

    order_folder_name = os.path.join(working_dir, "Order")
    config_folder_name = os.path.join(working_dir, "Temp_Data")
    config_file_path = os.path.join(config_folder_name, config_file_name)

    Path(order_folder_name).mkdir(parents=True, exist_ok=True)

    config = load_config(config_file_path)
    model_path = config['model_path']
    provider_uri = config['provider_uri']
    qlib.init(provider_uri=provider_uri, region="cn")

    model = load_model(model_path)
    dataset = prepare_data(config)
    pred_series = model.predict(dataset)
    pred_df = pred_series.to_frame("score")
    pred_df.index = pred_df.index.set_names(['datetime', 'instrument'])

    order_generator = GenerateOrder(provider_uri = provider_uri ,working_dir=order_folder_name, stock_basic_csv=basic_info_path, pred_df=pred_df)
    order_generator.generate_order_csv()

if __name__ == "__main__":
    main()