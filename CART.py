from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:
    """
    结点类，用于存储树的节点信息
    """

    def __init__(self, feature=None, left=None, right=None, *, label=None):
        self.feature = feature  # 分割特征的索引
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.label = label  # 叶子节点的标签

    # 判断是否为叶子节点
    def is_leaf_node(self):
        return self.label is not None


class DecisionTree:
    """
    决策树类，
    """

    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split  # 节点分裂所需的最少样本数
        self.max_depth = max_depth  # 树的最大深度
        self.root = None  # 树的根节点
        self.available_features = None

    # 开始训练决策树
    def fit(self, X, y):
        self.available_features = np.ones(X.shape[1], dtype=int)
        self.root = self._grow_tree(X, y)

    # 根据决策树进行预测
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _weighted_gini(self, y, X_column):
        y_left = y[X_column == 1]
        y_right = y[X_column == 0]

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)

        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return weighted_gini

    def _best_split(self, X, y):
        best_gini = 1
        best_feature = None

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        for feature in range(X.shape[1]):
            if self.available_features[feature]:
                current_gini = self._weighted_gini(y, X[:, feature])
                if current_gini < best_gini:
                    best_gini = current_gini
                    best_feature = feature

        return best_feature, best_gini < self._gini(y)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.label

        if x[node.feature] == 1:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _grow_tree(self, X, y, depth=0):
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

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # 生成子树
        self.available_features[best_feature] = 0
        left_indices = X[:, best_feature] == 1
        right_indices = ~left_indices

        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, left=left_child, right=right_child)


def calculate_accuracy(list1, list2):
    # 初始化不同的元素计数器
    difference_count = 0

    # 遍历列表中的每一个元素
    for i in range(len(list1)):
        # 如果元素不同，则增加计数器
        if list1[i] != list2[i]:
            difference_count += 1

    # 计算不同的比率
    difference_ratio = (len(list1) - difference_count) / len(list1)
    return difference_ratio


def main():
    # 加载数据集
    filename = 'lung_cancer_data.csv'
    dataset = pd.read_csv(filename)

    features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
                'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
                'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
    target = 'LUNG_CANCER'

    # 提取特征和目标变量
    X = dataset[features]
    y = dataset[target]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = DecisionTree()
    clf.fit(X_train, y_train)

    print(y_test.to_numpy())
    y_predict = clf.predict(X_test)
    print(y_predict)
    accuracy = calculate_accuracy(y_test.to_numpy(), y_predict)
    print("The accuracy of different elements is:", accuracy)


if __name__ == '__main__':
    main()
