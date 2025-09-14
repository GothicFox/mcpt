import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def donchian_breakout(ohlc: pd.DataFrame, lookback: int):
    # input df is assumed to have a 'close' column
    upper = ohlc['close'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['close'].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc['close'] > upper] = 1
    signal.loc[ohlc['close'] < lower] = -1
    signal = signal.ffill()
    return signal

def optimize_donchian(ohlc: pd.DataFrame):

    best_pf = 0
    best_lookback = -1
    r = np.log(ohlc['close']).diff().shift(-1)
    for lookback in range(12, 169):
        if lookback % 20 == 0:
            print(f"  Testing lookback {lookback}...")
        signal = donchian_breakout(ohlc, lookback)
        sig_rets = signal * r
        sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf

def walkforward_donch(ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 30):

    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None
    
    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_lookback, _ = optimize_donchian(ohlc.iloc[i-train_lookback:i])
            tmp_signal = donchian_breakout(ohlc, best_lookback)
            next_train += train_step
        
        wf_signal[i] = tmp_signal.iloc[i]
    
    return wf_signal

if __name__ == '__main__':

    # 读取 CSV 数据
    print("Loading data...")
    df = pd.read_csv('BTCUSDT_1h_merged.csv')

    # 将 timestamp 转换为 datetime 并设为索引
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
    df.set_index('timestamp', inplace=True)

    # 筛选 2017年8月到2020年8月的数据
    df = df[(df.index >= '2017-08-01') & (df.index < '2020-09-01')]
    print(f"Data shape: {df.shape}")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")

    print("Optimizing Donchian parameters...")
    best_lookback, best_real_pf = optimize_donchian(df)
    print(f"Best lookback: {best_lookback}, Best profit factor: {best_real_pf:.4f}")

    print("Generating signals...")
    signal = donchian_breakout(df, best_lookback) 

    df['r'] = np.log(df['close']).diff().shift(-1)
    df['donch_r'] = df['r'] * signal

    plt.style.use("dark_background")
    df['donch_r'].cumsum().plot(color='red')
    plt.title("In-Sample Donchian Breakout")
    plt.ylabel('Cumulative Log Return')
    plt.show()


