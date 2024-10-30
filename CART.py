import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv('lung_cancer_data.csv')
X = data.drop('target', axis=1)
y = data['target']

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


# 创建一个字典来存储每个元素及其出现次数
element_counts = {}

# 遍历二维列表
for sublist in X:
    # 获取每个一维列表的第一个元素
    first_element = sublist[0]
    # 使用字典的 get 方法来简化计数逻辑
    # get 方法会返回指定键的值，如果键不存在，则返回默认值（这里是0）
    element_counts[first_element] = element_counts.get(first_element, 0) + 1

# 输出结果
for element, count in element_counts.items():
    print(f"{element}: {count}")