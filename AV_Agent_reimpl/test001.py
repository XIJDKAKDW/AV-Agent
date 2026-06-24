# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pymysql
import joblib
from PIL import Image
import platform
import logging
from sklearn.feature_extraction.text import TfidfVectorizer as TF
system = platform.system()
from sklearn.feature_extraction.text import TfidfVectorizer

# 设置日志
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 将lambda函数定义为独立的函数，以便序列化
def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

def get_connection():
    host = "211.65.82.10"
    user = "root"
    password = 'chang123'
    db = 'malware_db'
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset='utf8',
        port = 3306,
    )
    return conn

# 1. MLP模型定义 (用于raw bytes特征)
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], num_classes=2):
        super(MLPModel, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

# 2. GNN模型定义 (用于控制流图特征)
class GNNModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=2):
        super(GNNModel, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index=None):
        # 简化的GNN实现，实际应用中应使用完整的图神经网络
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.classifier(x)
        return x

# 3. 传统机器学习模型 (用于Drebin特征)
class DrebinModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.vectorizer = None

    def fit(self, file_paths, y):
        # 使用TF-IDF向量化文件路径
        self.vectorizer = TF(
            input="filename",
            tokenizer=tokenizer_func,
            token_pattern=None,
            binary=True
        )
        X = self.vectorizer.fit_transform(file_paths)
        self.model.fit(X, y)
        return self

    def predict_proba(self, file_paths):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted")
        X = self.vectorizer.transform(file_paths)
        return self.model.predict_proba(X)

# 数据集类 - 对齐论文中的三种特征
class RawBytesDataset(Dataset):
    def __init__(self, seqs, conn):
        self.seqs = seqs
        self.conn = conn
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """从s3Feature表加载原始字节特征"""
        for seq in self.seqs:
            try:
                with self.conn.cursor() as cursor:
                    sql = "SELECT feature FROM s3_feature WHERE apkSeq = %s"
                    cursor.execute(sql, (seq,))
                    result = cursor.fetchone()

                    if result and result[0]:
                        feature_vector = [float(x.replace('[','').replace(']','')) for x in result[0].split(',')]
                        self.data.append(feature_vector)
                        label = self._get_label_by_seq(seq)
                        self.labels.append(label)
            except Exception as e:
                logger.error(f"Error loading raw bytes for seq {seq}: {e}")
                continue

    def _get_label_by_seq(self, seq):
        """根据seq获取标签"""
        try:
            with self.conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"Error getting label for seq {seq}: {e}")
        return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

class DrebinFeatureDataset:
    def __init__(self, seqs, conn, is_training=True, vectorizer=None):
        self.seqs = seqs
        self.conn = conn
        self.file_paths = []
        self.labels = []
        self.is_training = is_training
        self.vectorizer = vectorizer
        self._load_data()

    def _load_data(self):
        """从drebin_feature表加载文件路径"""
        for seq in self.seqs:
            try:
                with self.conn.cursor() as cursor:
                    sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
                    cursor.execute(sql, (seq,))
                    result = cursor.fetchone()

                    if result and result[0]:
                        file_path = result[0]
                        if system == "Linux":
                            pass
                        else:
                            file_path = file_path.replace('/home/changxiaosong/dataset', r'D:')
                        self.file_paths.append(file_path)

                        label = self._get_label_by_seq(seq)
                        self.labels.append(label)
                        #logger.info(f"Loaded Drebin seq {seq}: file_path={file_path}, label={label}")
            except Exception as e:
                logger.error(f"Error loading Drebin feature for seq {seq}: {e}")
                continue

    def _get_label_by_seq(self, seq):
        """根据seq获取标签"""
        try:
            with self.conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"Error getting label for seq {seq}: {e}")
        return 0

    def get_features_and_labels(self):
        """返回特征和标签"""
        if self.is_training:
            # 训练模式：创建并拟合TF-IDF向量化器
            self.vectorizer = TF(
                input="filename",
                tokenizer=tokenizer_func,
                token_pattern=None,
                binary=True
            )
            X = self.vectorizer.fit_transform(self.file_paths)
        else:
            # 测试模式：使用现有的向量化器转换数据
            if self.vectorizer is not None:
                X = self.vectorizer.transform(self.file_paths)
            else:
                logger.error("TF-IDF vectorizer not provided for test data")
                return np.array([]), np.array([])

        #logger.info(f"Drebin TF-IDF特征矩阵形状: {X.shape}")
        return X, np.array(self.labels)

    def get_vectorizer(self):
        """返回TF-IDF向量化器"""
        return self.vectorizer

    def get_file_paths(self):
        """返回文件路径"""
        return self.file_paths

class ImageFeatureDataset(Dataset):
    def __init__(self, seqs, conn):
        self.seqs = seqs
        self.conn = conn
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """从pic_feature表加载图像路径并提取特征"""
        for seq in self.seqs:
            try:
                with self.conn.cursor() as cursor:
                    sql = "SELECT path FROM pic_feature WHERE apkSeq = %s"
                    cursor.execute(sql, (seq,))
                    result = cursor.fetchone()

                    if result and result[0]:
                        image_path = result[0]
                        #小数据集位置
                        if system == "Linux":
                            # D:/pics_dir/ae855f49daa06be8e51f06476ce38d8b78108b4bde758c9feec20d676e4e61d2.jpg
                            image_path=image_path.replace(r'D:/pics_dir/','/home/changxiaosong/dataset/db_7_little/')
                        feature_vector = self._process_image(image_path)
                        if feature_vector is not None:
                            self.data.append(feature_vector)
                            label = self._get_label_by_seq(seq)
                            self.labels.append(label)
            except Exception as e:
                logger.error(f"Error loading image feature for seq {seq}: {e}")
                continue

    def _process_image(self, image_path):
        """处理图像并提取特征"""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return None

            img = Image.open(image_path).convert('RGB')
            img = img.resize((64, 64))
            img_array = np.array(img) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))
            return img_array

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def _get_label_by_seq(self, seq):
        """根据seq获取标签"""
        try:
            with self.conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"Error getting label for seq {seq}: {e}")
        return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# 训练函数 - 对齐论文中的层次特征构建
def train_raw_bytes_model(train_file):
    """训练原始字节分类器"""
    logger.info("开始训练原始字节分类器...")

    seqs_train = load_seqs_from_file(train_file)
    conn = get_connection()
    if conn is None:
        return None

    dataset = RawBytesDataset(seqs_train, conn)
    if len(dataset) == 0:
        logger.error("错误: 没有有效的原始字节数据")
        conn.close()
        return None

    input_dim = len(dataset[0][0])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPModel(input_dim=input_dim, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    conn.close()
    torch.save(model.state_dict(), 'raw_bytes_model.pth')
    logger.info("原始字节分类器训练完成")
    return model

def train_drebin_model(train_file):
    """训练Drebin特征分类器"""
    logger.info("开始训练Drebin特征分类器...")

    seqs_train = load_seqs_from_file(train_file)
    conn = get_connection()
    if conn is None:
        return None, None

    dataset = DrebinFeatureDataset(seqs_train, conn, is_training=True)
    X_train, y_train = dataset.get_features_and_labels()
    vectorizer = dataset.get_vectorizer()

    if X_train.shape[0] == 0:
        logger.error("错误: 没有有效的Drebin特征数据")
        conn.close()
        return None, None

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    conn.close()

    # 保存模型和向量化器
    joblib.dump(model, 'drebin_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    logger.info("Drebin特征分类器训练完成")
    return model, vectorizer

def train_image_model(train_file):
    """训练图像特征分类器"""
    logger.info("开始训练图像特征分类器...")

    seqs_train = load_seqs_from_file(train_file)
    conn = get_connection()
    if conn is None:
        return None

    dataset = ImageFeatureDataset(seqs_train, conn)
    if len(dataset) == 0:
        logger.error("错误: 没有有效的图像数据")
        conn.close()
        return None

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CNN模型定义
    class CNNModel(nn.Module):
        def __init__(self, num_classes=2):
            super(CNNModel, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8))
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = CNNModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    conn.close()
    torch.save(model.state_dict(), 'image_model.pth')
    logger.info("图像特征分类器训练完成")
    return model

def generate_detection_thresholds(train_file, test_file):
    """生成三类检测阈值并存入数据库 - 对齐论文的层次特征构建"""
    logger.info("开始生成检测阈值...")

    # 训练三个模型 - 对应论文中的层次特征
    raw_bytes_model = train_raw_bytes_model(train_file)
    drebin_model, vectorizer = train_drebin_model(train_file)
    image_model = train_image_model(train_file)

    if raw_bytes_model is None or drebin_model is None or image_model is None:
        logger.error("模型训练失败，无法生成检测阈值")
        return

    # 加载测试数据
    seqs_test = load_seqs_from_file(test_file)
    conn = get_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cursor:
            for seq in seqs_test:
                try:
                    # 获取原始字节预测概率
                    pro_raw_bytes = predict_raw_bytes_probability(raw_bytes_model, seq, conn)
                    # 获取Drebin特征预测概率
                    pro_drebin = predict_drebin_probability(drebin_model, vectorizer, seq, conn)
                    # 获取图像特征预测概率
                    pro_image = predict_image_probability(image_model, seq, conn)

                    # 插入到av_agent_feature表

                    sql = """
                    INSERT INTO av_agent_feature (apkSeq, pro_raw_bytes, pro_drebin, pro_image)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    pro_raw_bytes = VALUES(pro_raw_bytes),
                    pro_drebin = VALUES(pro_drebin),
                    pro_image = VALUES(pro_image)
                    """
                    cursor.execute(sql, (seq, pro_raw_bytes, pro_drebin, pro_image))

                except Exception as e:
                    logger.error(f"处理seq {seq}时出错: {e}")
                    continue

        conn.commit()
        logger.info("检测阈值生成完成并存入数据库")

    except Exception as e:
        logger.error(f"生成检测阈值时出错: {e}")
        conn.rollback()
    finally:
        conn.close()

def predict_raw_bytes_probability(model, seq, conn):
    """预测原始字节模型的恶意概率"""
    try:
        with conn.cursor() as cursor:
            sql = "SELECT feature FROM s3_feature WHERE apkSeq = %s"
            cursor.execute(sql, (seq,))
            result = cursor.fetchone()

            if result and result[0]:
                feature_vector = [float(x.replace('[','').replace(']','')) for x in result[0].split(',')]
                features_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)

                device = next(model.parameters()).device
                features_tensor = features_tensor.to(device)

                model.eval()
                with torch.no_grad():
                    output = model(features_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    malicious_prob = probabilities[0, 1].item()
                return f"{malicious_prob:.4f}"
    except Exception as e:
        logger.error(f"预测原始字节概率时出错 (seq {seq}): {e}")
    return "0.0000"

def predict_drebin_probability(model, vectorizer, seq, conn):
    """预测Drebin特征模型的恶意概率"""
    try:
        with conn.cursor() as cursor:
            sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
            cursor.execute(sql, (seq,))
            result = cursor.fetchone()

            if result and result[0]:
                file_path = result[0]
                if system == "Linux":
                    pass
                else:
                    file_path = file_path.replace('/home/changxiaosong/dataset', r'D:')

                # 使用TF-IDF向量化器转换特征
                features_tfidf = vectorizer.transform([file_path])

                # 预测概率
                probabilities = model.predict_proba(features_tfidf)
                malicious_prob = probabilities[0, 1]
                return f"{malicious_prob:.4f}"
    except Exception as e:
        logger.error(f"预测Drebin概率时出错 (seq {seq}): {e}")
    return "0.0000"

def predict_image_probability(model, seq, conn):
    """预测图像特征模型的恶意概率"""
    try:
        dataset = ImageFeatureDataset([seq], conn)
        if len(dataset) == 0:
            return "0.0000"

        image_tensor, _ = dataset[0]
        image_tensor = image_tensor.unsqueeze(0)

        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            malicious_prob = probabilities[0, 1].item()
        return f"{malicious_prob:.4f}"

    except Exception as e:
        logger.error(f"预测图像概率时出错 (seq {seq}): {e}")
    return "0.0000"

def load_seqs_from_file(file_path):
    """从文件加载序列号"""
    seqs = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    seqs.append(int(line))
    except Exception as e:
        logger.error(f"错误: 加载文件时出错: {e}")
    return seqs

def main(train_file = r"D:\研究生\日常文件备份\123.txt",test_file = r"D:\研究生\日常文件备份\123.txt"):
    logger.info("开始训练模型并生成检测阈值...")
    generate_detection_thresholds(train_file, test_file)
    logger.info("程序执行完成！")

if __name__ == "__main__":
    main()