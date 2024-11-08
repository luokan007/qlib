
import requests


def get_eastmoney():
    url = "http://4.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&fs=m:105,m:106,m:107&fields=f12"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError("request error")

    try:
        _symbols = [_v["f12"].replace("_", "-P") for _v in resp.json()["data"]["diff"].values()]
    except Exception as e:
        print(f"request error: {e}")
        raise

    if len(_symbols) < 8000:
        raise ValueError("request error")

    return sorted(_symbols)
symbol_list = get_eastmoney()
print(symbol_list)