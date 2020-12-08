def risk_metrics(data, lam, ignore_first=True):

    data = data.copy()**2
    last_ret = data.iloc[int(ignore_first)].copy()

    for i in range(1 + int(ignore_first), data.shape[0]):
        curr = data.iloc[i].copy()
        data.iloc[i] = lam*(data.iloc[i-1]) + (1-lam)*(last_ret)
        last_ret = curr

    return data