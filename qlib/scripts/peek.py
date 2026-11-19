"""Peek at the data through Qlib's expression engine (a short take for the video)."""
import qlib
from qlib.data import D


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/us_alpaca", region="us")

    members = D.instruments("sp500")
    for day in ("2016-01-04", "2020-03-16", "2026-11-17"):
        n = len(D.list_instruments(members, start_time=day, end_time=day, as_list=True))
        print(f"S&P 500 members on {day}: {n}")

    fields = ["$close", "Mean($close, 20) / $close", "Ref($close, -2) / Ref($close, -1) - 1"]
    names = ["close", "20d average / close", "label: next day return"]
    df = D.features(["AAPL", "NVDA", "SIVB"], fields, start_time="2023-03-06", end_time="2023-03-10")
    df.columns = names
    print()
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
