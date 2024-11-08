import qlib
from qlib.workflow import R

qlib.init()
exp_name = "combine"
rid = "df620dfb313c4531a804a922167e43e8"
predict_recorder = R.get_recorder(recorder_id=rid, experiment_name=exp_name)
pred_df = predict_recorder.load_object('pred.pkl')
print(pred_df)


##################

# import pickle
# with open(
#         r"G:\qlibrolling\mlruns\1\df620dfb313c4531a804a922167e43e8\artifacts\pred.pkl",
#         "rb") as f:
#     pred_df = pickle.load(f)
# print(pred_df)
