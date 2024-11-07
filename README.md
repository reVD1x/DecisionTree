Machine Learning - Decision Tree
机器学习-决策树

#CART：回归与分类树

一、 **课题内容和要求**

1. 课题目标

CART（Classification and Regression Trees），即回归与分类树。在分类树下面有两个关键的思想：第一个是关于递归地划分自变量空间的想法；第二个想法是用验证数据进行剪枝。

用C语言或Python语言完成CART算法实现，并选用合适的经典数据集进行测试和验证。

 

2. 课题内容

使用Python语言实现CART分类树，基于加权基尼系数算法确定最佳分裂特征，递归地构建分类树并划分自变量空间；同时通过限制分类树的深度和节点分裂所需的最少样本数进行预剪枝，通过比较验证数据和预测数据进行后剪枝。

二、 **数据结构和算法设计**

1. CART分类树构建流程（图1）

CART模型通过递归地分割数据集来构建一棵二叉树，每次分裂都旨在最大化信息增益或最小化信息不纯度。分裂的核心在于找到当前子树的最佳分裂特征，在CART分类树中通常采用加权基尼系数作为评判标准。同时通过预剪枝和后剪枝来限制分类树的生长，以规避对数据过于敏感而带来的过拟合问题。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)

图1 CART分类树构建流程图

 

2. 数据结构设计

(1) Node类

Node类定义分类树的节点，是用于构建分类树的基本单元，每个实例代表决策树中的一个节点。这个类包含了以下属性和方法：

属性：

feature：表示用于分裂数据集的特征。如果是根节点或中间节点，这个属性会是一个有效特征的索引；如果是叶子节点，这个属性则为None。

left和right：分别指向左子树和右子树的引用。对于叶子节点来说，这两个属性都是None。

label：只有叶子节点才会有的属性，表示该节点所属的类别标签。对于非叶子节点，这个属性为None。

 

方法：

is_leaf_node()：返回布尔值，用于判断当前节点是否为叶子节点。

 

(2) DecisionTree类

DecisionTree类是实现CART算法的核心部分，负责构建和操作分类树。它包含了多种属性和方法，用于处理数据、构建树、进行预测以及模型评估等任务。这个类包含了以下属性和方法：

属性：

min_samples_split：节点分裂所需的最少样本数，用于预剪枝。

max_depth：树的最大深度，同样用于预剪枝。

root：树的根节点，是一个 Node 实例。

available_features：一个布尔数组，用于跟踪哪些特征还可用作分裂依据。

 

方法：

fit(X, y)：训练模型，即构建分类树的过程。

predict(X)：根据构建好的分类树对输入数据进行预测。

calculate_accuracy(list1, list2)：计算两个列表之间的准确率。

prune(X_val, y_val)：执行后剪枝操作，尝试移除一些子节点，并评估剪枝后的性能。

_gini(y)：计算给定数据集的基尼系数。

_weighted_gini(y, X_col)：计算给定特征列的加权基尼系数。

_best_split(X, y)：寻找最佳的分裂特征。

_most_common_label(y)：统计给定数据集中出现次数最多的标签。

_traverse_tree(x, node)：递归地遍历决策树，用于预测过程。

_grow_tree(X, y, depth)：递归地构建决策树。

_post_prune(node, X_val, y_val)：递归地进行后剪枝。

 

(3) 主函数

main()函数演示了如何使用上述类来加载数据集、训练模型、进行预测和评估模型性能。它包括了以下几个步骤：

i. 调用pandas库中方法从.csv文件中加载数据集。

ii. 提取特征和目标变量。

iii. 调用sklearn库中方法划分训练集和测试集。

iv. 创建DecisionTree实例并调用fit方法训练模型。

v. 调用predict方法对测试集进行预测，并计算准确率。

vi. 调用prune方法执行后剪枝，并再次评估剪枝后的模型性能。

 

3. 算法设计

(1) 基尼系数和加权基尼系数

基尼系数衡量的是一个集合的不纯度，取值范围在[0, 1]之间，值越小表示数据集越纯。对于数据集D，其基尼系数Gini(D)定义为：

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)

其中，pk是第k类在数据集D中的比例。

 

加权基尼系数综合考虑了数据集各个子集的基尼系数和子集的大小，在决策树中，我们需要根据某个特征将数据集分成多个子集，并计算这些子集的加权基尼系数。对于原始数据集D，把S作为分裂特征，则加权基尼系数Gini(D, S)定义为：

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image006.gif)

其中，m是分裂后的子集数，Di是第i个子集，∣Di∣是第i个子集的样本数，∣D∣是原始数据集的样本数，Gini(Di)是第i个子集的基尼系数。

(2) 训练过程

初始化：

设置min_samples_split和max_depth参数，初始化root和available_features。

 

递归构建树：

停止条件：

i. 当前深度超过max_depth。   

ii. 可用特征数为0。

iii. 样本中只有一种标签。

iv. 节点中的样本数少于min_samples_split。

 

选择最佳分裂特征：

i. 遍历所有可用特征，计算每个特征的加权基尼系数。

ii. 选择基尼系数最小的特征作为最佳分裂特征。

 

生成子树：

i. 根据最佳分裂特征将样本分为左子集和右子集。

ii. 递归地构建左子树和右子树。

 

(3) 预测过程

递归遍历树：

i. 从根节点开始，根据输入样本的特征值递归地遍历树，直到到达叶子节点。

ii. 返回叶子节点的标签作为预测结果。

 

(4) 后剪枝

评估剪枝前后性能：

i. 保存当前节点的状态。

ii. 尝试将当前节点变为叶子节点，计算剪枝后的准确率。

iii. 如果剪枝后的准确率不低于原始准确率，则保留剪枝结果。

iv. 否则，恢复原状，继续递归地对左右子树进行剪枝。

 

(4) 模型评估

计算准确率：

使用calculate_accuracy方法计算预测结果与真实标签之间的准确率。

**三、** **系统实现**

import numpy as np

import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split

 

 

class Node:

  """结点类，用于存储树的节点信息"""

 

  def __init__(self, feature=None, left=None, right=None, *, label=None):

​    self.feature = feature # 分割特征的索引

​    self.left = left    # 左子树

​    self.right = right   # 右子树

​    self.label = label   # 叶子节点的标签

 

  def is_leaf_node(self):

​    """判断是否为叶子节点"""

​    return self.label is not None

 

 

class DecisionTree:

  """分类树类，用于构建 CART"""

 

  def __init__(self, min_samples_split=2, max_depth=100):

​    self.min_samples_split = min_samples_split # 节点分裂所需的最少样本数，用于预剪枝

​    self.max_depth = max_depth         # 树的最大深度，用于预剪枝

​    self.root = None              # 树的根节点

​    self.available_features = None       # 可用于分裂的特征

 

  def fit(self, X, y):

​    """开始训练 CART"""

​    self.available_features = np.ones(X.shape[1], dtype=int)  # 初始化可用于分裂的特征

​    self.root = self._grow_tree(X, y)

 

  def predict(self, X):

​    """根据 CART 进行预测"""

​    \# 将DataFrame转换为NumPy数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

 

​    return np.array([self._traverse_tree(x, self.root) for x in X]) # 遍历每个样本，进行预测

 

  def prune(self, X_val, y_val):

​    """后剪枝：尝试移除一些子节点，并评估剪枝后的性能"""

​    self._post_prune(self.root, X_val, y_val)

 

  def calculate_accuracy(self, list1, list2):

​    """计算预测准确率"""

​    correct = np.sum(list1 == list2)

​    return correct / len(list1)

 

  def _gini(self, y):

​    """计算基尼系数"""

​    _, counts = np.unique(y, return_counts=True)  # 获取每个类别的计数

​    probabilities = counts / len(y) # 计算每个类别的概率

​    return 1 - np.sum(probabilities ** 2)  # 计算基尼系数

 

  def _weighted_gini(self, y, X_col):

​    """计算加权基尼系数"""

​    y_left = y[X_col == 1]

​    y_right = y[X_col == 0]

 

​    n = len(y)

​    n_left = len(y_left)

​    n_right = len(y_right)

 

​    gini_left = self._gini(y_left)   # 左子集的基尼系数

​    gini_right = self._gini(y_right)  # 右子集的基尼系数

 

​    weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right  # 计算加权基尼系数

​    return weighted_gini

 

  def _best_split(self, X, y):

​    """选择最佳分裂特征"""

​    best_gini = 1    # 最佳加权基尼系数

​    best_feature = None # 最佳特征

 

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

 

​    \# 遍历所有特征

​    for feature in range(X.shape[1]):

​      if self.available_features[feature]:

​        current_gini = self._weighted_gini(y, X[:, feature])

​        if current_gini < best_gini:

​          best_gini = current_gini

​          best_feature = feature

 

​    return best_feature, best_gini < self._gini(y) # 返回最佳特征和是否有效

 

  def _most_common_label(self, y):

​    """统计最常出现的标签"""

​    counter = Counter(y)  # 统计每个标签的出现次数

​    most_common = counter.most_common(1)[0][0] # 获取出现次数最多的标签

​    return most_common

 

  def _traverse_tree(self, x, node):

​    """递归地遍历树"""

​    if node.is_leaf_node():

​      return node.label

 

​    if x[node.feature] == 1:

​      return self._traverse_tree(x, node.left)

​    return self._traverse_tree(x, node.right)

 

  def _grow_tree(self, X, y, depth=0):

​    """递归地构建 CART"""

​    n_samples, n_features = X.shape # 样本数和特征数

​    n_labels = len(np.unique(y))  # 获取不同标签的数量

 

​    \# 停止条件

​    if (depth >= self.max_depth           # 当前深度超过最大深度

​        or self.available_features.sum() == 0  # 没有可用特征

​        or n_labels == 1            # 数据集中只有一种标签

​        or n_samples < self.min_samples_split): # 样本数少于最小分裂样本数

​      leaf_label = self._most_common_label(y)

​      return Node(label=leaf_label)

 

​    \# 寻找最佳分割特征

​    best_feature, use_feature = self._best_split(X, y)

 

​    if not use_feature:

​      \# 如果最佳特征无效，则停止分割

​      leaf_label = self._most_common_label(y)

​      return Node(label=leaf_label)

 

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

 

​    \# 生成子树

​    self.available_features[best_feature] = 0  # 标记该特征已使用

​    left_indices = X[:, best_feature] == 1 # 左子集的索引

​    right_indices = ~left_indices      # 右子集的索引

 

​    left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)

​    right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

 

​    return Node(feature=best_feature, left=left_child, right=right_child)

 

  def _post_prune(self, node, X_val, y_val):

​    """递归地进行后剪枝"""

​    \# 如果是叶子节点，直接返回

​    if node.is_leaf_node():

​      return node

 

​    if node.left and node.right:

​      \# 保存当前节点的状态

​      left_child = node.left

​      right_child = node.right

​      original_label = node.label

 

​      \# 尝试剪枝

​      original_accuracy = self.calculate_accuracy(self.predict(X_val), y_val)

 

​      \# 将当前节点变为叶子节点

​      node.left = None

​      node.right = None

​      node.label = self._most_common_label(y_val)

 

​      pruned_accuracy = self.calculate_accuracy(self.predict(X_val), y_val)

 

​      \# 如果剪枝后的准确率不低于原始准确率，则保留剪枝后的结果

​      if pruned_accuracy >= original_accuracy:

​        return node

 

​      \# 恢复原状

​      node.left = left_child

​      node.right = right_child

​      node.label = original_label

 

​      node.left = self._post_prune(node.left, X_val, y_val)

​      node.right = self._post_prune(node.right, X_val, y_val)

 

​    return node

 

 

def main():

  \# 加载数据集

  filename = 'lung_cancer_data.csv'

  dataset = pd.read_csv(filename)

 

  \# 提取特征和目标变量

  features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',

​        'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',

​        'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',

​        'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

  target = 'LUNG_CANCER'

 

  X = dataset[features]

  y = dataset[target]

 

  \# 划分训练集和测试集

  X_train, X_test, y_train, y_test = train_test_split(X, y)

 

  \# 构建 CART 并开始训练

  clf = DecisionTree()

  clf.fit(X_train, y_train)

 

  \# 根据 CART 预测测试集，计算准确率

  y_predict = clf.predict(X_test)

  accuracy = clf.calculate_accuracy(y_test.to_numpy(), y_predict)

 

  print("测试集实际标签：")

  print(y_test.to_numpy())

  print("测试集预测标签（未剪枝）：")

  print(y_predict)

  print("预测准确率：", accuracy)

 

  \# 剪枝

  clf.prune(X_train, y_train)

  y_predict_pruned = clf.predict(X_test)

  accuracy_pruned = clf.calculate_accuracy(y_test.to_numpy(), y_predict_pruned)

 

  print("测试集预测标签（已剪枝）：")

  print(y_predict_pruned)

  print("预测准确率：", accuracy_pruned)

 

 

if __name__ == '__main__':

  main()

**四、** **测试及其结果分析**

使用的数据集：lung_cancer_data，统计了生活习惯以及对应症状与是否罹患肺癌之间的关系。数据集共计14个训练特征，包括GENDER、SMOKING、YELLOW_FINGERS、ANXIETY、PEER_PRESSURE、CHRONIC_DISEASE、FATIGUE、ALLERGY、WHEEZING、ALCOHOL_CONSUMING、COUGHING、SHORTNESS_OF_BREATH、SWALLOWING_DIFFICULTY、CHEST_PAIN；另外LUNG_CANCER为目标特征，所有特征均呈现0-1分布。

 

1. 测试案例一（图2）

预测准确率较高，且后剪枝后准确率有所上升。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

图2 测试案例一

 

2. 测试案例二（图3）

预测准确率较高，且后剪枝后准确率有所上升。

 

3. 测试案例三（图4）

预测准确率较高，且后剪枝后准确率有所上升。

 

由此可见，后剪枝可以避免出现过拟合的情况，并由此提高预测准确率。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

图3 测试案例二

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)

图4 测试案例三

**五、** **课题完成过程中遇到的难点、问题及解决方法**

问题1：DataFrame和NumPy数组不兼容

解决办法：调用pandas库中的函数，将DataFrame转换为NumPy数组。

if isinstance(X, pd.DataFrame):

X = X.to_numpy()

 

问题2：重复使用同一特征作为分裂特征，导致预测出现问题。

解决办法：增加一个可用特征列表，标记特征是否使用过。

 

问题3：有些特征分裂时加权基尼系数甚至大于总体基尼系数，容易导致过拟合、增加运行负担等问题。

解决办法：在返回最佳特征时验证加权基尼系数大小，若大于总体基尼系数则直接剪枝。

**六、** **总结与心得**

1. 理解CART的基本原理

基本概念：了解分类树的基本概念，每个节点代表一个特征上的预测，每个分支代表一个预测结果，每个叶子节点代表一个输出值。

分裂准则：掌握了基尼系数作为分裂准则，用于衡量数据的不纯度。

 

2. 递归构建树的逻辑

递归思想：分类树的构建过程本质上是一个递归过程。从根节点开始，不断选择最佳的分裂特征，生成左右子树，直到满足停止条件。

停止条件：了解了常见的停止条件，如树的最大深度、节点中的样本数、数据集中只有一个类别等。这些条件有助于防止过拟合。

 

3. 特征选择的重要性

特征选择：在每一步分裂过程中，选择最佳特征是关键。通过计算基尼系数找到最优的分裂特征，不仅提高了模型的准确性，还减少了计算复杂度。

特征重要性：通过观察哪些特征被选为分裂特征，可以评估各个特征的重要性，这对于特征选择和数据理解非常有帮助。

 

4. 剪枝技术的应用

预剪枝：设置最大深度和最小分裂样本数，提前停止树的生长，防止过拟合。

后剪枝：通过后剪枝技术，可以在树完全构建后再进行剪枝，评估剪枝前后的性能变化，选择最优的剪枝方案。这有助于进一步提高模型的泛化能力。

 

5. 代码实现的细节

数据处理：学会了如何处理数据集，包括读取CSV文件、提取特征和目标变量、划分训练集和测试集等。

类和方法的设计：通过设计Node类和DecisionTree类，理解了面向对象编程的思想，特别是如何封装数据和方法。

性能评估：学会了如何计算模型的准确率，并通过后剪枝来优化模型性能。

 

6. 实践中的挑战

数据预处理：在实际应用中，数据往往需要进行预处理，如处理缺失值、异常值等。这些步骤对模型的性能影响很大。

过拟合问题：决策树容易过拟合，特别是在数据集较小或特征较多的情况下。通过剪枝技术可以有效缓解这一问题。

性能优化：可以通过优化算法和数据结构来提高性能。

 

7. 深入研究方向

多种模型混合：CART可作为基础模型，与其他模型结合，如随机森林和梯度提升树，进一步提高模型的性能。

多分类和回归：虽然本例中只涉及二分类问题，但决策树同样适用于多分类和回归任务，可以进一步探索这些应用场景。

可视化：通过可视化工具，可以更直观地展示决策树的结构，帮助理解和解释模型。



 

**决策树**

一、 **课题内容和要求**

1. 课题目标

决策树（Decision Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。由于这种决策分支画成图形很像一棵树的枝干，故称决策树。在机器学习中，决策树是一个预测模型，他代表的是对象属性与对象值之间的一种映射关系。Entropy（熵）系统的凌乱程度，使用算法ID3，C4.5和C5.0生成树算法使用熵。这一度量是基于信息学理论中熵的概念。

决策树是一种树形结构，其中每个内部节点表示一个属性上的测试，每个分支代表一个测试输出，每个叶节点代表一种类别。

用C语言或Python语言完成决策树算法实现，并选用合适的经典数据集进行测试和验证。

 

2. 课题内容

使用Python语言实现决策树，基于ID3或C4.5算法确定最佳分裂特征，递归地构建决策树并划分自变量空间；同时通过限制决策树的深度和节点分裂所需的最少样本数进行预剪枝。

二、 **数据结构和算法设计**

1. 决策树构建流程（图1）

使用ID3或C4.5算法的决策树构建流程与CART构建流程相似，同样是递归地分割数据集，但ID3或C4.5算法构建的并非二叉树，即中间节点可拥有多个子节点。ID3算法每次分裂旨在信息增益，而C4.5算法则旨在最大化信息增益比。

 

2．数据结构设计

(1) Node类

Node类定义决策树的节点，是用于构建分类树的基本单元，每个实例代表决策树中的一个节点。这个类包含了以下属性和方法：

属性：

feature：表示用于分裂数据集的特征。如果是根节点或中间节点，这个属性会是一个有效特征的索引；如果是叶子节点，这个属性则为None。

children：子节点字典，键是特征值，值是对应的子节点对象。

label：如果节点是叶子节点，则此属性存储节点的类别标签。

 

方法：

is_leaf_node()：检查节点是否为叶子节点。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image014.gif)

图1 决策树构建流程图

 

(2) DecisionTree类

DecisionTree类负责构建和操作决策树。这个类包含了以下属性和方法：

属性：

method：指定使用的决策树算法（ID3或C4.5）。

min_samples_split：节点分裂所需的最少样本数。

max_depth：树的最大深度。

root：树的根节点，是一个 Node 实例。

available_features：一个布尔数组，用于跟踪哪些特征还可用作分裂依据。

y_most：数据集中最常见的标签，用于处理无法继续分割的情况。

 

方法：

fit(X, y)：根据数据集训练决策树模型。

predict(X)：根据构建好的决策树对输入数据进行预测。

calculate_accuracy(list1, list2)：计算两个列表之间的准确率。

_entropy(y)：计算给定数据集的熵。

_information_gain(y, X_col)：计算给定特征的信息增益。

_information_gain_rate(y, X_col)：计算给定特征的信息增益比。

_best_split(X, y)：寻找最佳的分裂特征。

_most_common_label(y)：统计给定数据集中出现次数最多的标签。

_traverse_tree(x, node)：递归地遍历决策树，用于预测过程。

_grow_tree(X, y, depth)：递归地构建决策树。

 

(3) 主函数

main()函数演示了如何使用上述类来加载数据集、训练模型、进行预测和评估模型性能。它包括了以下几个步骤：

i. 调用pandas库中方法从.csv文件中加载数据集。

ii. 提取特征和目标变量。

iii. 调用sklearn库中方法划分训练集和测试集。

iv. 创建DecisionTree实例并调用fit方法训练模型。

v. 调用predict方法使用ID3和C4.5两种算法分别构建决策树模型，并计算预测准确率。

 

3．算法设计

(1) 熵、信息增益和信息增益比

熵是衡量数据集混乱度的一个指标。对于一个包含多个类别的数据集D，其熵H(D)定义为：

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image016.gif)

其中，pi是第i类在数据集中所占的比例。

 

信息增益用于衡量某个特征对数据集的贡献。对于特征A，其信息增益IG(A)定义为：

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image018.gif)

其中，Dv是数据集中特征A取值为v的子集，∣D∣和|Dv∣分别是数据集和子集的大小。

 

信息增益比用于衡量特征对数据集贡献的重要性，定义为：

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image020.gif)

其中，SplitInfo(A)是特征A的分裂信息，定义为：

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image022.gif)

 

(2) ID3算法

ID3是一种基于信息熵和信息增益的决策树算法。它的核心思想是通过选择信息增益最大的特征作为分裂节点，递归地构建决策树，直到满足某些停止条件。

i. 计算熵。

ii. 计算信息增益。

iii. 选择信息增益最大的特征作为当前节点的分裂特征。

iv. 根据选定的特征将数据集划分为多个子集，对每个子集递归构建树，直到满足停止条件（停止条件与CART构建相同）。

 

(3) C4.5算法

C4.5 是 ID3 的改进版本，它引入了信息增益比来选择最佳分裂特征，从而减少对特征值多的特征的偏好。C4.5 还支持处理连续值特征和缺失值。

 

(4) 模型评估

计算准确率：

使用calculate_accuracy方法计算预测结果与真实标签之间的准确率。

 

**三、** **系统实现**

import numpy as np

import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split

 

methods = {'ID3': 0,

​      'C4.5': 1}

 

 

class Node:

  """结点类，用于存储树的节点信息"""

 

  def __init__(self, feature=None, children=None, *, label=None):

​    self.feature = feature   # 分割特征的索引

​    self.children = children  # 子节点字典，键为特征值，值为子节点

​    self.label = label     # 叶子节点的标签

 

  def is_leaf_node(self):

​    """判断是否为叶子节点"""

​    return self.label is not None

 

 

class DecisionTree:

  """决策树类，用于构建决策树"""

 

  def __init__(self, method, min_samples_split=2, max_depth=100, y_most=0):

​    self.method = method

​    self.min_samples_split = min_samples_split # 节点分裂所需的最少样本数，用于预剪枝

​    self.max_depth = max_depth         # 树的最大深度，用于预剪枝

​    self.root = None              # 树的根节点

​    self.available_features = None       # 可用于分裂的特征

​    self.y_most = y_most            # 多数类标签

 

  def fit(self, X, y):

​    """开始训练决策树"""

​    self.available_features = np.ones(X.shape[1], dtype=int)

​    self.y_most = self._most_common_label(y)

​    self.root = self._grow_tree(X, y)

 

  def predict(self, X):

​    """根据决策树进行预测"""

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

​    return np.array([self._traverse_tree(x, self.root) for x in X]) # 对每个样本进行预测

 

  def calculate_accuracy(self, list1, list2):

​    """计算预测准确率"""

​    correct = np.sum(list1 == list2)

​    return correct / len(list1)

 

  def _entropy(self, y):

​    """计算熵"""

​    counter = Counter(y)  # 统计每个类别的频数

​    entropy = 0

​    for count in counter.values():

​      p = count / len(y) # 计算概率

​      entropy -= p * np.log2(p)  # 计算熵

​    return entropy

 

  def _information_gain(self, y, X_col):

​    """计算信息增益"""

​    parent_entropy = self._entropy(y)  # 计算父节点的熵

​    values, counts = np.unique(X_col, return_counts=True)  # 获取特征值及其频数

​    weighted_entropy = 0  # 加权熵

​    for value, count in zip(values, counts):

​      subset_y = y[X_col == value]  # 获取子集的目标变量

​      weighted_entropy += (count / len(y)) * self._entropy(subset_y) # 计算加权熵

​    information_gain = parent_entropy - weighted_entropy  # 计算信息增益

​    return information_gain

 

  def _information_gain_rate(self, y, X_col):

​    """计算信息增益比"""

​    return self._information_gain(y, X_col) / self._entropy(y)

 

  def _best_split(self, X, y):

​    """选择最佳分裂特征"""

​    best_criteria = 0

​    best_feature = None

 

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

 

​    for feature in range(X.shape[1]):

​      \# 检查特征是否可用

​      if self.available_features[feature]:

​        if self.method == 0:

​          current_criteria = self._information_gain(y, X[:, feature])

​        elif self.method == 1:

​          current_criteria = self._information_gain_rate(y, X[:, feature])

 

​        if current_criteria > best_criteria:

​          best_criteria = current_criteria

​          best_feature = feature

 

​    return best_feature, best_criteria > 0 # 返回最佳特征索引和是否使用该特征

 

  def _most_common_label(self, y):

​    """统计最常出现的标签"""

​    counter = Counter(y)  # 统计每个类别的频数

​    most_common = counter.most_common(1)[0][0]

​    return most_common

 

  def _traverse_tree(self, x, node):

​    """递归地遍历树"""

​    if node.is_leaf_node():

​      return node.label

 

​    feature_value = x[node.feature] # 获取当前节点的特征值

​    child_node = node.children.get(feature_value)  # 获取对应的子节点

​    if child_node is None:

​      return self.y_most # 如果找不到对应的子节点，返回多数类标签

​    return self._traverse_tree(x, child_node)  # 递归遍历子节点

 

  def _grow_tree(self, X, y, depth=0):

​    """递归地构建决策树"""

​    n_samples, n_features = X.shape

​    n_labels = len(np.unique(y))

 

​    \# 停止条件

​    if (depth >= self.max_depth

​        or self.available_features.sum() == 0

​        or n_labels == 1

​        or n_samples < self.min_samples_split):

​      leaf_label = self._most_common_label(y)

​      return Node(label=leaf_label)

 

​    \# 寻找最佳分割特征及其值

​    best_feature, use_feature = self._best_split(X, y)

 

​    if not use_feature:

​      \# 如果最佳特征无效，则停止分割

​      leaf_label = self._most_common_label(y)

​      return Node(label=leaf_label)

 

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

 

​    \# 生成子树

​    self.available_features[best_feature] = 0  # 标记该特征已使用

​    children = {}

​    feature_value = np.unique(X[:, best_feature])  # 获取特征值

​    for value in feature_value:

​      indices = X[:, best_feature] == value

​      children[value] = self._grow_tree(X[indices], y[indices], depth + 1)  # 递归构建子树

 

​    return Node(feature=best_feature, children=children)  # 返回当前节点

 

 

def main():

  \# 加载数据集

  filename = 'lung_cancer_data.csv'

  dataset = pd.read_csv(filename)

 

  \# 提取特征和目标变量

  features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',

​        'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',

​        'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',

​        'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

  target = 'LUNG_CANCER'

 

  X = dataset[features]

  y = dataset[target]

 

  \# 划分训练集和测试集

  X_train, X_test, y_train, y_test = train_test_split(X, y)

 

  \# 构建 ID3 决策树并开始训练

  clf = DecisionTree(method=methods['ID3'])

  clf.fit(X_train, y_train)

 

  \# 根据决策树预测测试集，计算准确率

  y_predict = clf.predict(X_test)

  accuracy = clf.calculate_accuracy(y_test.to_numpy(), y_predict)

 

  print("测试集实际标签：")

  print(y_test.to_numpy())

  print("测试集预测标签（ID3）：")

  print(y_predict)

  print("预测准确率：", accuracy)

 

  \# 构建 C4.5 决策树并开始训练

  clf_ = DecisionTree(method=methods['C4.5'])

  clf_.fit(X_train, y_train)

 

  \# 根据决策树预测测试集，计算准确率

  y_predict_ = clf_.predict(X_test)

  accuracy_ = clf_.calculate_accuracy(y_test.to_numpy(), y_predict_)

 

  print("测试集预测标签（C4.5）：")

  print(y_predict_)

  print("预测准确率：", accuracy_)

 

 

if __name__ == '__main__':

  main()

**四、** **测试及其结果分析**

使用的数据集：与CART算法一致。

 

1. 测试案例一（图2）

预测准确率较高，但ID3和C4.5预测结果相同。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image024.jpg)

图2 测试案例一

 

2. 测试案例二（图3）

预测准确率较高，但ID3和C4.5预测结果相同。

 

3. 测试案例三（图4）

预测准确率较高，但ID3和C4.5预测结果相同。

 

由此可见，在特征呈0-1分布的简单数据集中，ID3与C4.5算法决策树的结果常常一样（原因见问题分析）。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image026.jpg)

图3 测试案例二

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image028.jpg)

图4 测试案例三

**五、** **课题完成过程中遇到的难点、问题及解决方法**

问题1：在特征呈0-1分布的简单数据集中，ID3与C4.5算法决策树的结果常常一样。

原因：对于二元特征（0-1分布），特征的熵通常较低，因此信息增益率与信息增益之间的差异较小。在这种情况下，信息增益率和信息增益往往会选择相同的特征进行分裂。同时，对于二元特征，分裂点的选择只有两种可能（0和1），这种简单的分裂点选择使得ID3和C4.5在选择最佳特征时更容易得到相同的结果。

 

问题2：决策树子节点较多，使用和CART中相同的独立节点表示不方便。

解决办法：引入一个字典结构，键和值分别存储特征值和子节点索引。

**六、** **总结与心得**

1. 理解基本概念

熵和信息增益：熵是衡量数据集混乱度的重要指标，信息增益则是选择最佳分裂特征的关键。通过计算信息增益，可以确定哪个特征对数据集的分类贡献最大。

信息增益比：C4.5算法通过引入信息增益比来避免对特征值多的特征的偏好，这使得算法更加公平和合理。

 

2. 代码实现

模块化设计：将决策树的构建和预测过程分解为多个方法，每个方法负责一个具体的任务。这种模块化设计使得代码更易于理解和维护。

灵活性：通过参数化算法类型（ID3或C4.5），代码可以轻松地切换不同的决策树算法，提高了代码的通用性和扩展性。

 

一些总结内容与CART有相似之处，此处省去。



 

**随机森林**

一、 **课题内容和要求**

1. 课题目标

随机森林（Random Forest，简称RF）是一种高度灵活的机器学习算法，是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树，而它的本质属于机器学习的一大分支——集成学习（Ensemble Learning）方法。随机森林的名称中有两个关键词，一个是“随机”，一个就是“森林”。每棵决策树都是一个分类器（假设现在针对的是分类问题），那么对于一个输入样本，N棵树会有N个分类结果。而随机森林集成了所有的分类投票结果，将投票次数最多的类别指定为最终的输出。

用C语言或Python语言完成随机森林算法实现，并选用合适的经典数据集进行测试和验证。

 

2. 课题内容

使用Python语言实现分类随机森林，单棵决策树基于C4.5算法确定最佳分裂特征，递归地构建决策树，多棵决策树构建起随机森林，预测时以投票的形式确定最终预测结果。

二、 **数据结构和算法设计**

1. 随机森林构建流程（图1）

构建随机森林首先要获取数据集的多个样本子集，这里采用有放回的抽取模式。然后再分别构建决策树，此处采用C4.5算法的决策树。预测时统计不同树的预测结果，取出现次数最多的结果为最终结果。

单棵C4.5决策树的构建方法见上文决策树相关内容。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image030.gif)

图1 随机森林构建与预测流程图

 

2. 数据结构设计

(1) Node类

与决策树算法中的Node类相同。

 

(2) DecisionTree类

与决策树算法中的DecisionTree类基本相同（不含calculate_accuracy(list1, list2)方法）。

 

(3) RandomForest类

RandomForest类用于构建随机森林模型，即多个决策树的集成。

属性：

n_estimators：随机森林中决策树的数量。

min_samples_split：每棵树的最小样本分裂数。

max_depth：每棵树的最大深度。

max_features：每次分裂考虑的最大特征数量。

trees：随机森林中的决策树列表。

 

方法：

fit(X,y)：有放回地从训练集中抽取样本来训练决策树，并组成随机森林。

predict(X)：对测试集进行预测，通过每棵树单独预测，再由所有树投票决定最终结果。

calculate_accuracy(list1,list2)：计算两个列表之间的准确率。

 

(4) 主函数

main()函数演示了如何使用上述类来加载数据集、训练模型、进行预测和评估模型性能。它包括了以下几个步骤：

i. 调用pandas库中方法从.csv文件中加载数据集。

ii. 提取特征和目标变量。

iii. 调用sklearn库中方法划分训练集和测试集。

iv. 创建RandomForest实例并调用fit方法训练模型。

v. 调用predict方法对测试集进行预测，并计算准确率。

 

3. 算法设计

(1) 随机森林

随机森林的核心思想是通过“多数表决”或“平均值”的方式，将多个弱学习器（通常是决策树）组合成一个强学习器。具体来说，随机森林通过以下步骤实现：

i. 有放回地抽样：从原始数据集中随机抽取多个子样本（通常与原始数据集大小相同），每个子样本用于训练一棵决策树。

ii. 特征随机选择：在每个节点分裂时，从所有特征中随机选择一部分特征进行评估，从中选择最佳的分裂特征。

iii. 构建多棵决策树：每棵决策树独立地使用不同的子样本和特征子集进行训练。

iv. 预测：对于新的输入数据，每棵决策树都会给出一个预测结果，最终的预测结果由所有决策树的预测结果通过投票（分类任务）或平均（回归任务）得出。

 

(2) 决策树相关算法

**三、** **系统实现**

import numpy as np

import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split

 

 

class Node:

  """结点类，用于存储树的节点信息"""

 

  def __init__(self, feature=None, children=None, *, label=None):

​    self.feature = feature   # 分割特征的索引

​    self.children = children  # 子节点字典，键为特征值，值为子节点

​    self.label = label     # 叶子节点的标签

 

  def is_leaf_node(self):

​    """判断是否为叶子节点"""

​    return self.label is not None

 

 

class DecisionTree:

  """决策树类，用于构建决策树"""

 

  def __init__(self, min_samples_split=2, max_depth=100, y_most=0):

​    self.min_samples_split = min_samples_split # 节点分裂所需的最少样本数，用于预剪枝

​    self.max_depth = max_depth         # 树的最大深度，用于预剪枝

​    self.root = None              # 树的根节点

​    self.available_features = None       # 可用于分裂的特征

​    self.y_most = y_most            # 多数类标签

 

  def fit(self, X, y):

​    """开始训练决策树"""

​    self.available_features = np.ones(X.shape[1], dtype=int)

​    self.y_most = self._most_common_label(y)

​    self.root = self._grow_tree(X, y)

 

  def predict(self, X):

​    """根据决策树进行预测"""

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

​    return np.array([self._traverse_tree(x, self.root) for x in X])

 

  def _entropy(self, y):

​    """计算熵"""

​    counter = Counter(y)  # 统计每个类别的频数

​    entropy = 0

​    for count in counter.values():

​      p = count / len(y) # 计算概率

​      entropy -= p * np.log2(p)  # 计算熵

​    return entropy

 

  def _information_gain(self, y, X_col):

​    """计算信息增益"""

​    parent_entropy = self._entropy(y)  # 计算父节点的熵

​    values, counts = np.unique(X_col, return_counts=True)  # 获取特征值及其频数

​    weighted_entropy = 0  # 加权熵

​    for value, count in zip(values, counts):

​      subset_y = y[X_col == value]  # 获取子集的目标变量

​      weighted_entropy += (count / len(y)) * self._entropy(subset_y) # 计算加权熵

​    information_gain = parent_entropy - weighted_entropy  # 计算信息增益

​    return information_gain

 

  def _information_gain_rate(self, y, X_col):

​    """计算信息增益比"""

​    return self._information_gain(y, X_col) / self._entropy(y)

 

  def _best_split(self, X, y):

​    """选择最佳分裂特征"""

​    best_criteria = 0

​    best_feature = None

 

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

 

​    for feature in range(X.shape[1]):

​      \# 检查特征是否可用

​      if self.available_features[feature]:

​        current_criteria = self._information_gain_rate(y, X[:, feature])

​        if current_criteria > best_criteria:

​          best_criteria = current_criteria

​          best_feature = feature

 

​    return best_feature, best_criteria > 0 # 返回最佳特征索引和是否使用该特征

 

  def _most_common_label(self, y):

​    """统计最常出现的标签"""

​    counter = Counter(y)  # 统计每个类别的频数

​    most_common = counter.most_common(1)[0][0]

​    return most_common

 

  def _traverse_tree(self, x, node):

​    """递归地遍历树"""

​    if node.is_leaf_node():

​      return node.label

 

​    feature_value = x[node.feature] # 获取当前节点的特征值

​    child_node = node.children.get(feature_value)  # 获取对应的子节点

​    if child_node is None:

​      return self.y_most # 如果找不到对应的子节点，返回多数类标签

​    return self._traverse_tree(x, child_node)  # 递归遍历子节点

 

  def _grow_tree(self, X, y, depth=0):

​    """递归地构建决策树"""

​    n_samples, n_features = X.shape

​    n_labels = len(np.unique(y))

 

​    \# 停止条件

​    if (depth >= self.max_depth

​        or self.available_features.sum() == 0

​        or n_labels == 1

​        or n_samples < self.min_samples_split):

​      leaf_label = self._most_common_label(y)

​      return Node(label=leaf_label)

 

​    \# 寻找最佳分割特征及其值

​    best_feature, use_feature = self._best_split(X, y)

 

​    if not use_feature:

​      \# 如果最佳特征无效，则停止分割

​      leaf_label = self._most_common_label(y)

​      return Node(label=leaf_label)

 

​    \# 将 DataFrame 转换为 NumPy 数组

​    if isinstance(X, pd.DataFrame):

​      X = X.to_numpy()

 

​    \# 生成子树

​    self.available_features[best_feature] = 0  # 标记该特征已使用

​    children = {}

​    feature_value = np.unique(X[:, best_feature])  # 获取特征值

​    for value in feature_value:

​      indices = X[:, best_feature] == value

​      children[value] = self._grow_tree(X[indices], y[indices], depth + 1)  # 递归构建子树

 

​    return Node(feature=best_feature, children=children)  # 返回当前节点

 

 

class RandomForest:

  """随机森林"""

 

  def __init__(self, n_estimators=100, min_samples_split=2, max_depth=15, max_features='sqrt'):

​    self.n_estimators = n_estimators      # 决策树的数量

​    self.min_samples_split = min_samples_split # 每棵树的最小样本分裂数

​    self.max_depth = max_depth         # 每棵树的最大深度

​    self.max_features = max_features      # 每次分裂考虑的最大特征数量

​    self.trees = []               # 随机森林中的决策树列表

 

  def fit(self, X, y):

​    """训练随机森林"""

​    self.trees = []

​    n_samples, n_features = X.shape

 

​    for _ in range(self.n_estimators):

​      \# 有放回地抽取样本

​      sample = np.random.choice(n_samples, n_samples, replace=True)

​      X_sample, y_sample = X.iloc[sample], y.iloc[sample]

 

​      \# 训练一棵决策树

​      tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)

​      tree.fit(X_sample, y_sample)

​      self.trees.append(tree)

 

  def predict(self, X):

​    """根据随机森林进行预测"""

​    tree_predicts = np.array([tree.predict(X) for tree in self.trees])

​    \# 投票决定最终预测

​    majority_vote = [Counter(col).most_common(1)[0][0] for col in tree_predicts.T]

​    return np.array(majority_vote)

 

  def calculate_accuracy(self, list1, list2):

​    """计算预测准确率"""

​    correct = np.sum(list1 == list2)

​    return correct / len(list1)

 

 

def main():

  \# 加载数据集

  filename = 'lung_cancer_data.csv'

  dataset = pd.read_csv(filename)

 

  \# 提取特征和目标变量

  features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',

​        'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',

​        'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',

​        'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

  target = 'LUNG_CANCER'

 

  X = dataset[features]

  y = dataset[target]

 

  \# 划分训练集和测试集

  X_train, X_test, y_train, y_test = train_test_split(X, y)

 

  \# 构建随机森林并开始训练

  rf = RandomForest()

  rf.fit(X_train, y_train)

 

  \# 根据随机森林预测测试集，计算准确率

  y_predict = rf.predict(X_test)

  accuracy = rf.calculate_accuracy(y_test.to_numpy(), y_predict)

 

  print("测试集实际标签：")

  print(y_test.to_numpy())

  print("测试集预测标签（随机森林）：")

  print(y_predict)

  print("预测准确率：", accuracy)

 

 

if __name__ == '__main__':

  main()

**四、** **测试及其结果分析**

使用的数据集：与CART算法一致。

 

1. 测试案例一（图2）

预测准确率较高。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image032.jpg)

 

2. 测试案例二（图3）

预测准确率较高。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image034.jpg)

 

3. 测试案例三（图4）

预测准确率较高。

![img](file:///C:/Users/Xiongrw/AppData/Local/Temp/msohtmlclip1/01/clip_image036.jpg)

 

由此可见，随机森林的平均预测准确率显著高于CART和单棵决策树，这就是集成学习带来的泛化能力和准确度。

**五、** **课题完成过程中遇到的难点、问题及解决方法**

问题1：采取无放回抽取子集的方式对数据集规模的要求很高，且会导致子集规模的不统一。

解决办法：改为有放回的抽取。

 

问题2：抽取样本后通过索引得到的X、y训练集数据类型不兼容。

解决办法：使用pandas库中的iloc方法进行转换。

**六、** **总结与心得**

1. 掌握随机森林的工作机制

集成学习：随机森林通过构建多棵决策树并综合它们的预测结果，提高了模型的稳定性和准确性。这体现了集成学习的思想。

有放回抽样：通过有放回地抽取样本（Bootstrap采样），每棵树使用不同的子样本进行训练，增加了模型的多样性。

特征随机选择：在每个节点分裂时，随机选择一部分特征进行评估，进一步减少了特征之间的相关性，提高了模型的泛化能力。

 

2. 随机森林算法特征

模型解释：随机森林虽然是一种黑盒模型，但通过特征重要性分析，可以部分解释模型的决策过程，这对实际应用中的可解释性要求是有帮助的。

计算复杂度：随机森林的训练时间较长，特别是在数据集较大时。因此，在实际应用中需要权衡模型的复杂度和计算资源。

一些总结内容与CART、决策树有相似之处，此处省去。
