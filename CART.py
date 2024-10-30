import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
filename = 'lung_cancer_data.csv'
dataset = pd.read_csv(filename)

print(dataset)

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
