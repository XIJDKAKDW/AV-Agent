import platform
import joblib
import shap
import numpy as np
import torch
from typing import Dict, List, Tuple
import json
import logging
import sys

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/combine_compare_tool_method")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/ganerate_pic_graph")

from combine_compare_tool_method import get_connection

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

def get_feature_file_label_by_seq(conn, seqs):
    """根据序列号获取文件路径和标签"""
    file_paths = []
    labels = []

    for seq in seqs:
        try:
            with conn.cursor() as cursor:
                # 获取文件路径
                sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()

                if result and result[0]:
                    file_path = result[0]
                    if platform.system() != "Linux":
                        file_path = file_path.replace('/home/changxiaosong/dataset', r'D:')
                    file_paths.append(file_path)

                    # 获取标签
                    sql = "SELECT label FROM app_label WHERE seq = %s"
                    cursor.execute(sql, (seq,))
                    label_result = cursor.fetchone()
                    if label_result:
                        labels.append(0 if label_result[0] == 'B' else 1)
                    else:
                        labels.append(0)
        except Exception as e:
            logger.error(f"获取序列 {seq} 的文件路径和标签失败: {e}")
            continue

    return file_paths, labels

class FeatureEngineering:
    """特征工程模块 - 实现论文中的字符特征提取"""

    def __init__(self, drebin_model, feature_vectorizer, device=None):
        """
        初始化特征工程模块
        drebin_model: 训练好的Drebin GBDT模型
        feature_vectorizer: TF-IDF向量化器
        """
        self.drebin_model = drebin_model
        self.feature_vectorizer = feature_vectorizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_key_features_with_shap(self, file_paths: List[str], labels: List[int], top_k: int = 10) -> Dict:
        """
        使用SHAP提取关键特征
        返回: 包含关键特征信息的字典
        """
        logger.info("使用SHAP提取关键特征...")

        # 转换特征 - 保持为稀疏矩阵格式
        X = self.feature_vectorizer.transform(file_paths)
        feature_names = self.feature_vectorizer.get_feature_names_out()

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(self.drebin_model)

        # 将数据转换为适合SHAP的格式
        if hasattr(X, "toarray"):
            X_array = X.toarray()
        else:
            X_array = X

        # 计算SHAP值
        shap_values = explainer.shap_values(X_array)

        key_features = {
            'global_importance': self._get_global_feature_importance(explainer, X_array, feature_names),
            'sample_specific': {}
        }

        # 为每个样本提取关键特征
        for i, (file_path, label) in enumerate(zip(file_paths, labels)):
            sample_key_features = self._get_sample_key_features(
                shap_values, X_array, i, feature_names, top_k
            )
            key_features['sample_specific'][file_path] = {
                'key_features': sample_key_features,
                'true_label': label,
                'predicted_label': self.drebin_model.predict(X_array[i:i+1])[0],
                'confidence': np.max(self.drebin_model.predict_proba(X_array[i:i+1]))
            }

        return key_features

    def _get_sample_key_features(self, shap_values, X_array, sample_idx: int,
                                 feature_names: List[str], top_k: int = 10) -> List[Dict]:
        """
        获取单个样本的关键特征
        """
        # 处理多分类情况
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # 使用恶意类别的SHAP值
        else:
            shap_vals = shap_values

        # 获取当前样本的SHAP值
        sample_shap = shap_vals[sample_idx]

        # 获取特征值
        if hasattr(X_array, "iloc"):
            feature_values = X_array.iloc[sample_idx].values
        else:
            feature_values = X_array[sample_idx]

        # 创建特征重要性列表
        feature_importance = []
        for j in range(len(feature_names)):
            if feature_values[j] != 0:  # 只考虑存在的特征
                shap_val = sample_shap[j]
                feature_importance.append({
                    'feature_name': feature_names[j],
                    'shap_value': float(shap_val),
                    'feature_value': float(feature_values[j]),
                    'feature_type': self._categorize_feature(feature_names[j]),
                    'semantic_description': self._get_semantic_description(feature_names[j])
                })

        # 按SHAP值的绝对值排序，取前top_k个
        feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return feature_importance[:top_k]

    def _get_global_feature_importance(self, explainer, X_array, feature_names, top_k: int = 20) -> List[Tuple]:
        """获取全局特征重要性"""
        # 使用模型自带的特征重要性
        if hasattr(self.drebin_model, 'feature_importances_'):
            importances = self.drebin_model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_k]
            return [(feature_names[i], float(importances[i])) for i in indices]

        # 备用方法：使用SHAP的均值绝对值
        try:
            shap_vals = explainer.shap_values(X_array)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # 恶意类别
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            indices = np.argsort(mean_abs_shap)[::-1][:top_k]
            return [(feature_names[i], float(mean_abs_shap[i])) for i in indices]
        except Exception as e:
            logger.error(f"全局特征重要性计算失败: {e}")
            return []

    def _categorize_feature(self, feature_name: str) -> str:
        """对Drebin特征进行分类"""
        feature_lower = feature_name.lower()

        if 'permission' in feature_lower:
            return 'permission'
        elif 'api' in feature_lower:
            return 'api'
        elif 'activity' in feature_lower:
            return 'activity'
        elif 'service' in feature_lower:
            return 'service'
        elif 'receiver' in feature_lower:
            return 'receiver'
        elif 'provider' in feature_lower:
            return 'provider'
        elif 'intent' in feature_lower:
            return 'intent'
        elif 'url' in feature_lower or 'domain' in feature_lower:
            return 'network'
        elif 'hardware' in feature_lower:
            return 'hardware'
        else:
            return 'other'

    def _get_semantic_description(self, feature_name: str) -> str:
        """Get semantic description of the feature"""
        feature_lower = feature_name.lower()

        # Permission-related features
        if 'permission' in feature_lower:
            perm_name = feature_name.split('permissionslist_')[-1] if 'permissionslist_' in feature_lower else feature_name
            return f"Requested permission: {perm_name}"

        # API-related features
        elif 'api' in feature_lower:
            api_name = feature_name.split('apilist_')[-1] if 'apilist_' in feature_lower else feature_name
            return f"API call: {api_name}"

        # Component-related features
        elif any(comp in feature_lower for comp in ['activity', 'service', 'receiver', 'provider']):
            comp_type = 'Activity' if 'activity' in feature_lower else \
                'Service' if 'service' in feature_lower else \
                    'BroadcastReceiver' if 'receiver' in feature_lower else 'ContentProvider'
            comp_name = feature_name.split('list_')[-1]
            return f"{comp_type} component: {comp_name}"

        # Network-related features
        elif 'url' in feature_lower or 'domain' in feature_lower:
            url = feature_name.split('urldomainlist_')[-1] if 'urldomainlist_' in feature_lower else feature_name
            return f"Network connection: {url}"

        else:
            return f"Feature: {feature_name}"
class LLMFeatureFormatter:
    """格式化特征用于LLM推理"""

    @staticmethod
    def format_confidence_scores(confidence_scores: Dict) -> str:
        """Format confidence scores"""
        formatted = "Classifier Confidence Scores:\n"
        for feature_type, scores in confidence_scores.items():
            formatted += f"- {feature_type}: {scores['confidence']:.4f} (Prediction: {'Malicious' if scores['predicted_class'] == 1 else 'Benign'})\n"
        return formatted
    @staticmethod
    def format_key_features(key_features: Dict) -> str:
        """Format key features"""
        formatted = "Key Semantic Feature Analysis:\n"

        # Group features by type
        features_by_type = {}
        for feature_info in key_features:
            feature_type = feature_info['feature_type']
            if feature_type not in features_by_type:
                features_by_type[feature_type] = []
            features_by_type[feature_type].append(feature_info)

        # Output features by type
        for feature_type, features in features_by_type.items():
            formatted += f"\n{feature_type.upper()} FEATURES:\n"
            for feat in features[:5]:  # Show max 5 features per type
                influence = "Positive" if feat['shap_value'] > 0 else "Negative"
                formatted += f"  • {feat['semantic_description']} (Influence: {influence}, Strength: {abs(feat['shap_value']):.4f})\n"

        return formatted
    @staticmethod
    def generate_structured_prompt(confidence_scores: Dict, key_features: Dict,
                                   sample_info: Dict = None) -> str:
        """生成用于LLM推理的结构化提示"""
        prompt = """
As a professional malware analysis expert, please analyze the maliciousness of this sample based on the following characteristics:

{confidence_section}

{feature_section}

Analysis Requirements:
1. Evaluate the maliciousness indication strength of each feature
2. Analyze potential malicious behavior patterns from feature combinations
3. Provide final judgment considering all evidence comprehensively
4. Provide brief reasoning process

Output Format:
Analysis Result: [Benign/Malicious]
Confidence Level: [High/Medium/Low]
Key Evidence: [List the 2-3 most significant features]
Reasoning Process: [Briefly explain the analysis logic]
""".format(
            confidence_section=LLMFeatureFormatter.format_confidence_scores(confidence_scores),
            feature_section=LLMFeatureFormatter.format_key_features(key_features)
        )

        return prompt

# 集成到主流程中的函数
def extract_features_for_llm(seqs: List[int], drebin_model, feature_vectorizer,
                             conn, llm_formatter: LLMFeatureFormatter = None) -> Dict:
    """
    完整的特征工程流程：从序列号提取特征并格式化为LLM输入
    """
    if llm_formatter is None:
        llm_formatter = LLMFeatureFormatter()

    # 获取文件路径和标签
    file_paths, labels = get_feature_file_label_by_seq(conn, seqs)

    # 初始化特征工程
    feature_engineer = FeatureEngineering(drebin_model, feature_vectorizer)

    results = {}

    # 提取关键特征
    key_features_data = feature_engineer.extract_key_features_with_shap(file_paths, labels)

    for i, seq in enumerate(seqs):
        try:
            # 获取该样本的置信度分数
            confidence_scores = get_classifier_confidence_scores(seq, conn)

            # 获取该样本的关键特征
            file_path = file_paths[i]
            if file_path in key_features_data['sample_specific']:
                sample_features = key_features_data['sample_specific'][file_path]['key_features']
            else:
                sample_features = []

            # 生成结构化文本
            structured_text = llm_formatter.generate_structured_prompt(
                confidence_scores, sample_features
            )

            results[seq] = {
                'confidence_scores': confidence_scores,
                'key_features': sample_features,
                'structured_text': structured_text,
                'true_label': labels[i],
                'file_path': file_path
            }

            logger.info(f"序列 {seq} 特征提取完成")

        except Exception as e:
            logger.error(f"序列 {seq} 特征提取失败: {e}")
            continue
    return results

def get_classifier_confidence_scores(seq: int, conn) -> Dict:
    confidence_scores = {}

    try:
        with conn.cursor() as cursor:
            # 查询av_agent_feature表
            sql = """
            SELECT pro_raw_bytes, pro_drebin, pro_image 
            FROM av_agent_feature 
            WHERE apkSeq = %s
            """
            cursor.execute(sql, (seq,))
            result = cursor.fetchone()

            if result:
                pro_raw_bytes, pro_drebin, pro_image = result

                # 处理原始字节特征置信度
                # if pro_raw_bytes and pro_raw_bytes != 'NULL':
                #     try:
                #         raw_bytes_confidence = float(pro_raw_bytes)
                #         confidence_scores['raw_bytes'] = {
                #             'confidence': raw_bytes_confidence,
                #             'predicted_class': 1 if raw_bytes_confidence >= 0.5 else 0
                #         }
                #     except (ValueError, TypeError):
                #         logger.warning(f"原始字节置信度格式错误: {pro_raw_bytes}")

                # 处理Drebin特征置信度
                if pro_drebin and pro_drebin != 'NULL':
                    try:
                        drebin_confidence = float(pro_drebin)
                        confidence_scores['drebin'] = {
                            'confidence': drebin_confidence,
                            'predicted_class': 1 if drebin_confidence >= 0.5 else 0
                        }
                    except (ValueError, TypeError):
                        logger.warning(f"Drebin置信度格式错误: {pro_drebin}")

                # 处理图像特征置信度
                # if pro_image and pro_image != 'NULL':
                #     try:
                #         image_confidence = float(pro_image)
                #         confidence_scores['image'] = {
                #             'confidence': image_confidence,
                #             'predicted_class': 1 if image_confidence >= 0.5 else 0
                #         }
                #     except (ValueError, TypeError):
                #         logger.warning(f"图像特征置信度格式错误: {pro_image}")

                logger.info(f"序列 {seq} 置信度分数获取成功:")
                for model_name, scores in confidence_scores.items():
                    logger.info(f"    - {model_name}: {scores['confidence']:.4f} (预测: {scores['predicted_class']})")

            else:
                logger.warning(f"未找到序列 {seq} 在av_agent_feature表中的记录")

    except Exception as e:
        logger.error(f"获取序列 {seq} 置信度分数时出错: {e}")

    return confidence_scores

# 使用示例
def main(test_seqs = [27141,26506,92417,26184,109850,26705,14548,27342,12659,25750]):
    # 加载训练好的模型和向量化器
    drebin_model = joblib.load('drebin_model.pkl')
    feature_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # 获取数据库连接
    conn = get_connection()

    # 提取特征
    results = extract_features_for_llm(test_seqs, drebin_model, feature_vectorizer, conn)

    # 保存结果
    with open('llm_features.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印示例
    for seq, result in list(results.items())[:1]:
        print(f"\n序列 {seq} 的结构化文本:")
        print(result['structured_text'])

    conn.close()

if __name__ == "__main__":
    main()