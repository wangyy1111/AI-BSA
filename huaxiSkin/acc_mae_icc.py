import numpy as np
 
def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mae -- MAE 评价指标
    """
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae

import numpy as np

def icc(Y, icc_type="icc(3,1)"):
    """
    Args:
        Y: 待计算的数据
        icc_type: 共支持 icc(2,1), icc(2,k), icc(3,1), icc(3,k)四种
    """

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    # MSC = SSC / dfc / n
    MSC = SSC / dfc

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc(2,1)" or icc_type == 'icc(2,k)':
        if icc_type=='icc(2,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    elif icc_type == "icc(3,1)" or icc_type == 'icc(3,k)':
        if icc_type=='icc(3,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC

a = [
     [0.65,0.18,0.6567,0.0883,0.22,0.05,0.12,0.0125,0.0187,0.03,0.0057,0.053,0.045,0.092,0.0083,0.0177,0.0717,0.1017,0.0267,0.2567,0.026,0.5167],
     [0.7082,0.1733,0.7082,0.1260,0.1430,0.0797,0.1661,0.0257,0.0386,0.0575,0.0276,0.0312,0.0981,0.1632,0.0671,0.0549,0.1352,0.1689,0.0761,0.3218,0.0873,0.5879]
     ]
b = np.array(a)
c = icc(b.T, icc_type="icc(2,1)")
print(c) 

y_true = np.array( [0.65,0.18,0.6567,0.0883,0.22,0.05,0.12,0.0125,0.0187,0.03,0.0057,0.053,0.045,0.092,0.0083,0.0177,0.0717,0.1017,0.0267,0.2567,0.026,0.5167])
y_pred = np.array( [0.7082,0.1733,0.7082,0.1260,0.1430,0.0797,0.1661,0.0257,0.0386,0.0575,0.0276,0.0312,0.0981,0.1632,0.0671,0.0549,0.1352,0.1689,0.0761,0.3218,0.0873,0.5879])
print (mae_value(y_true, y_pred))
