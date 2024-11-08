import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# 算法类型
methods = {'ID3': 0,
           'C4.5': 1}


class Node:
    """结点类，用于存储树的节点信息"""

    def __init__(self, feature=None, children=None, *, label=None):
        self.feature = feature      # 分割特征的索引
        self.children = children    # 子节点字典，键为特征值，值为子节点
        self.label = label          # 叶子节点的标签

    def is_leaf_node(self):
        """判断是否为叶子节点"""
        return self.label is not None


class DecisionTree:
    """决策树类，用于构建决策树"""

    def __init__(self, method, min_samples_split=2, max_depth=100, y_most=0):
        self.method = method
        self.min_samples_split = min_samples_split  # 节点分裂所需的最少样本数，用于预剪枝
        self.max_depth = max_depth                  # 树的最大深度，用于预剪枝
        self.root = None                            # 树的根节点
        self.available_features = None              # 可用于分裂的特征
        self.y_most = y_most                        # 多数类标签

    def fit(self, X, y):
        """开始训练决策树"""
        self.available_features = np.ones(X.shape[1], dtype=int)
        self.y_most = self._most_common_label(y)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """根据决策树进行预测"""
        # 将 DataFrame 转换为 NumPy 数组
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self._traverse_tree(x, self.root) for x in X]) # 对每个样本进行预测

    def calculate_accuracy(self, list1, list2):
        """计算预测准确率"""
        correct = np.sum(list1 == list2)
        return correct / len(list1)

    def _entropy(self, y):
        """计算熵"""
        counter = Counter(y)    # 统计每个类别的频数
        entropy = 0
        for count in counter.values():
            p = count / len(y)  # 计算概率
            entropy -= p * np.log2(p)   # 计算熵
        return entropy

    def _information_gain(self, y, X_col):
        """计算信息增益"""
        parent_entropy = self._entropy(y)   # 计算父节点的熵
        values, counts = np.unique(X_col, return_counts=True)   # 获取特征值及其频数
        weighted_entropy = 0    # 加权熵
        for value, count in zip(values, counts):
            subset_y = y[X_col == value]    # 获取子集的目标变量
            weighted_entropy += (count / len(y)) * self._entropy(subset_y)  # 计算加权熵
        information_gain = parent_entropy - weighted_entropy    # 计算信息增益
        return information_gain

    def _information_gain_rate(self, y, X_col):
        """计算信息增益比"""
        return self._information_gain(y, X_col) / self._entropy(y)

    def _best_split(self, X, y):
        """选择最佳分裂特征"""
        best_criteria = 0
        best_feature = None

        # 将 DataFrame 转换为 NumPy 数组
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        for feature in range(X.shape[1]):
            # 检查特征是否可用
            if self.available_features[feature]:
                if self.method == 0:
                    current_criteria = self._information_gain(y, X[:, feature])
                elif self.method == 1:
                    current_criteria = self._information_gain_rate(y, X[:, feature])

                if current_criteria > best_criteria:
                    best_criteria = current_criteria
                    best_feature = feature

        return best_feature, best_criteria > 0  # 返回最佳特征索引和是否使用该特征

    def _most_common_label(self, y):
        """统计最常出现的标签"""
        counter = Counter(y)    # 统计每个类别的频数
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _traverse_tree(self, x, node):
        """递归地遍历树"""
        if node.is_leaf_node():
            return node.label

        feature_value = x[node.feature] # 获取当前节点的特征值
        child_node = node.children.get(feature_value)   # 获取对应的子节点
        if child_node is None:
            return self.y_most  # 如果找不到对应的子节点，返回多数类标签
        return self._traverse_tree(x, child_node)   # 递归遍历子节点

    def _grow_tree(self, X, y, depth=0):
        """递归地构建决策树"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 停止条件
        if (depth >= self.max_depth
                or self.available_features.sum() == 0
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_label = self._most_common_label(y)
            return Node(label=leaf_label)

        # 寻找最佳分割特征及其值
        best_feature, use_feature = self._best_split(X, y)

        if not use_feature:
            # 如果最佳特征无效，则停止分割
            leaf_label = self._most_common_label(y)
            return Node(label=leaf_label)

        # 将 DataFrame 转换为 NumPy 数组
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # 生成子树
        self.available_features[best_feature] = 0   # 标记该特征已使用
        children = {}
        feature_value = np.unique(X[:, best_feature])   # 获取特征值
        for value in feature_value:
            indices = X[:, best_feature] == value
            children[value] = self._grow_tree(X[indices], y[indices], depth + 1)    # 递归构建子树

        return Node(feature=best_feature, children=children)    # 返回当前节点


def main():
    # 加载数据集
    filename = 'lung_cancer_data.csv'
    dataset = pd.read_csv(filename)

    # 提取特征和目标变量
    features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
                'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
                'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
    target = 'LUNG_CANCER'

    X = dataset[features]
    y = dataset[target]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 构建 ID3 决策树并开始训练
    clf = DecisionTree(method=methods['ID3'])
    clf.fit(X_train, y_train)

    # 根据决策树预测测试集，计算准确率
    y_predict = clf.predict(X_test)
    accuracy = clf.calculate_accuracy(y_test.to_numpy(), y_predict)

    print("测试集实际标签：")
    print(y_test.to_numpy())
    print("测试集预测标签（ID3）：")
    print(y_predict)
    print("预测准确率：", accuracy)

    # 构建 C4.5 决策树并开始训练
    clf_ = DecisionTree(method=methods['C4.5'])
    clf_.fit(X_train, y_train)

    # 根据决策树预测测试集，计算准确率
    y_predict_ = clf_.predict(X_test)
    accuracy_ = clf_.calculate_accuracy(y_test.to_numpy(), y_predict_)

    print("测试集预测标签（C4.5）：")
    print(y_predict_)
    print("预测准确率：", accuracy_)


if __name__ == '__main__':
    main()
