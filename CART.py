import pandas as pd
from sklearn.model_selection import train_test_split


# 加载数据集
filename = 'lung_cancer_data.csv'
dataset = pd.read_csv(filename)

features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
            'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ',
            'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
            'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
target = 'LUNG_CANCER'

# 提取特征和目标变量
X = dataset[features]
y = dataset[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

def calculate_gini(groups, classes):
    # 总共的实例数量
    n_instances = float(sum([len(group) for group in groups]))
    print(n_instances)

    # 初始化基尼指数为0
    gini = 0.0

    # 对每一个子集进行遍历
    for group in groups:
        # 如果子集为空，则跳过
        if len(group) == 0:
            continue

        # 计算该子集中每个类别的概率
        score = 0.0
        for class_val in classes:
            # 计算属于该类别的实例数量
            p = [row[-1] for row in group].count(class_val) / len(group)

            # 加上该类别概率的平方
            score += p * p

        # 对每个子集的纯度加权求和，得到总的基尼指数
        gini += (1.0 - score) * (len(group) / n_instances)

    return gini


