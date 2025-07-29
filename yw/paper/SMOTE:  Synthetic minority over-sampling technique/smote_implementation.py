import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X_minority, N=100, k=5, random_state=None):
    """
    SMOTE算法实现
    参数：
    - X_minority: 少数类样本 (ndarray) [n_minority, n_features]
    - N: 需要生成的新样本百分比（如 200 表示增加两倍）
    - k: 近邻个数
    - random_state: 随机种子

    返回：
    - synthetic_samples: 合成的新样本 (ndarray)
    """
    if random_state:
        np.random.seed(random_state)
    
    n_minority, n_features = X_minority.shape
    
    # N<100时，假设为50，表示每个样本要生成0.5个样本点，在生成的代码中是不合法的，所以需要预处理：
    # 对原样本随机选择N/100个样本，再对这些样本进行N=100的SMOTE
    if N < 100:
        # 对样本进行随机选择再过采样
        n_samples = int(n_minority * N / 100)
        indices = np.random.choice(n_minority, n_samples, replace=False)
        X_minority = X_minority[indices]
        N = 100

    N = int(N / 100)  # 表示每个样本要合成的数量

    # NearestNeighbors用于实现最近邻搜索，n_neighbors是希望找出每个样本k个最近邻，fit函数传入少数样本构建最近邻模型
    neigh = NearestNeighbors(n_neighbors=k).fit(X_minority)
    # kneighbors函数用于查询每个样本的最近邻
    neighbors = neigh.kneighbors(return_distance=False)

    synthetic_samples = []
    
    # 遍历每个样本
    for i in range(X_minority.shape[0]):
        # 每个样本生成N个数据
        for n in range(N):
            nn_index = np.random.randint(1, k)  # 跳过自己（index 0）
            neighbor = X_minority[neighbors[i][nn_index]]

            # 在样本点和选定的邻居之间随机插入一个新点，diff为两点的向量差，gap为随机权重[0,1)，synthetic是合成的新样本点
            diff = neighbor - X_minority[i]
            gap = np.random.rand()
            synthetic = X_minority[i] + gap * diff
            synthetic_samples.append(synthetic)
    
    return np.array(synthetic_samples)