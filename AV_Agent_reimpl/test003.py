# coding=utf-8
import json
import logging
import os
import platform
import sys
from typing import Dict, List

import joblib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/combine_compare_tool_method")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/ganerate_pic_graph")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test002")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_3")
from pr2_new_3.test001Method_new_9_4_3 import llm_chat

def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

from combine_compare_tool_method import get_connection
from test002 import extract_features_for_llm, get_classifier_confidence_scores
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class TwoPhaseReasoningEngine:
    """
    两阶段推理引擎 - 实现论文中的两阶段推理机制
    对应论文IV-C节: Two-Phase Reasoning
    """

    def __init__(self, llm_config: Dict = None):
        self.llm_config = llm_config or {
            "base_url": "http://211.65.82.10:8087/api/generate",
            "model": "codellama:13b",
            "timeout": 60
        }

    def execute_phase1_reasoning(self, confidence_scores: Dict) -> Dict:
        """
        第一阶段推理 - 基于置信度分数的分析
        对应论文中的 R1_LLM : T ! R1
        """
        logger.info("执行第一阶段推理")

        prompt = self._build_phase1_prompt(confidence_scores)
        llm_response = self._call_llm(prompt, phase="phase1")

        parsed_result = self._parse_phase1_response(llm_response)

        return {
            'prompt': prompt,
            'llm_response': llm_response,
            'parsed_result': parsed_result
        }

    def execute_phase2_reasoning(self, phase1_result: Dict, key_features: Dict, formatted: Dict = None) -> Dict:
        """
        第二阶段推理 - 结合关键字符特征的详细分析
        对应论文中的 R2_LLM : R1 × ET3 ! A

        Args:
            phase1_result: 第一阶段推理结果
            key_features: 关键特征
            formatted: 包含raw_bytes和image信息的格式化数据
        """
        logger.info("执行第二阶段推理")

        prompt = self._build_phase2_prompt(phase1_result, key_features, formatted)
        llm_response = self._call_llm(prompt, phase="phase2")
        parsed_result = self._parse_phase2_response(llm_response)

        return {
            'prompt': prompt,
            'llm_response': llm_response,
            'parsed_result': parsed_result
        }

    def _build_phase1_prompt(self, confidence_scores: Dict) -> str:
        """Build phase 1 reasoning prompt"""
        formatted_scores = self._format_confidence_scores(confidence_scores)

        return f"""
As a professional malware analysis expert, please conduct a preliminary analysis based on the following classifier confidence scores:

## Classifier Confidence Scores:
{formatted_scores}

## Analysis Requirements (Phase 1):
1. Evaluate the confidence levels and consistency of each classifier
2. Analyze the maliciousness tendency reflected by the confidence scores
3. Consider the weights of different feature types (Drebin features typically have more semantic significance)
4. Provide preliminary maliciousness judgment and main evidence

## Output Format (Strictly follow this format):
[Phase 1 Analysis]
Preliminary Judgment: [Benign/Suspicious/Malicious]
Main Evidence: [Analysis based on confidence scores, indicate which classifier provides the strongest evidence]
Confidence Assessment: [High/Medium/Low]
Reasoning Summary: [Briefly explain the analysis logic, no more than 100 words]
"""

    def _build_phase2_prompt(self, phase1_result: Dict, key_features: Dict, formatted: Dict = None) -> str:
        """Build phase 2 reasoning prompt with enhanced feature information"""
        formatted_features = self._format_key_features(key_features, formatted)
        phase1_response = phase1_result.get('llm_response', 'Phase 1 analysis results unavailable')

        return f"""
As a professional malware analysis expert, please make a final judgment based on Phase 1 analysis and key feature characteristics:

## Phase 1 Analysis Results:
{phase1_response}

## Key Feature Characteristics Analysis:

### Raw Bytes Key Features (Reflecting Binary Patterns):
{formatted_features['raw_bytes']}

### Drebin Key Features (Reflecting Program Behavior):
{formatted_features['drebin']}

### Image Key Features (Reflecting Visual Information):
{formatted_features['image']}

## Analysis Requirements (Phase 2):
1. Combine key feature characteristics to validate or revise Phase 1 judgment
2. Analyze malicious behavior patterns reflected by feature combinations
3. Identify the most indicative malicious features (especially Drebin features)
4. Provide final classification and detailed reasoning process

## Output Format (Strictly follow this format):
[Phase 2 Analysis]
Final Classification: [Benign/Malicious]
Key Evidence: [List the 2-3 most important features and their maliciousness indications]
Behavior Pattern: [Identified malicious behavior patterns, such as mining, ransomware, etc.]
Detailed Reasoning: [Complete analysis reasoning process, explaining how conclusions are drawn from features]
Final Confidence: [Comprehensive confidence based on all evidence: High/Medium/Low]
"""

    def _format_confidence_scores(self, confidence_scores: Dict) -> str:
        """Format confidence scores"""
        if not confidence_scores:
            return "No valid confidence data"

        formatted = ""
        for feature_type, scores in confidence_scores.items():
            confidence = scores.get('confidence', 0)
            predicted_class = scores.get('predicted_class', 0)
            description = scores.get('description', feature_type)

            risk_level = "High risk" if confidence > 0.7 else "Medium risk" if confidence > 0.5 else "Low risk"
            pred_label = "Malicious" if predicted_class == 1 else "Benign"

            formatted += f"- {description}: {confidence:.4f} ({risk_level}, Prediction: {pred_label})\n"

        # 添加统计摘要
        if confidence_scores:
            conf_values = [scores.get('confidence', 0) for scores in confidence_scores.values()]
            avg_confidence = sum(conf_values) / len(conf_values)
            max_confidence = max(conf_values)

            formatted += f"\nStatistical Summary:\n"
            formatted += f"- Average Confidence: {avg_confidence:.4f}\n"
            formatted += f"- Highest Confidence: {max_confidence:.4f}\n"
            formatted += f"- Number of Classifiers: {len(confidence_scores)}\n"
        return formatted

    def _format_key_features(self, key_features: Dict, formatted: Dict = None) -> Dict:
        """格式化关键特征，包含raw_bytes和image信息"""
        formatted_result = {
            'raw_bytes': "No significant features",
            'drebin': "No significant features",
            'image': "No significant features"
        }

        # 处理SHAP关键特征
        key_features_convert = {}
        for one in key_features:
            one_feature = {}
            feature_type = one['feature_type']
            tmp = key_features_convert.get(feature_type, [])

            one_feature['description'] = one['semantic_description']
            one_feature['importance'] = one['shap_value']
            one_feature['feature_type'] = one['feature_type']
            tmp.append(one_feature)
            key_features_convert[feature_type] = tmp

        # 格式化Drebin特征
        drebin_text = ""
        for key in key_features_convert.keys():
            features = key_features_convert[key][:5]
            for i, feat in enumerate(features[:5], 1):
                desc = feat.get('description', 'Unknown feature')
                importance = feat.get('importance', 0)
                drebin_text += f"{i}. {desc} (Importance: {importance:.4f})\n"

        if drebin_text:
            formatted_result['drebin'] = drebin_text

        # 处理formatted中的raw_bytes和image信息
        if formatted:
            # 处理raw_bytes信息
            if formatted.get('raw_bytes'):
                raw_bytes_info = self._extract_raw_bytes_info(formatted['raw_bytes'])
                formatted_result['raw_bytes'] = raw_bytes_info
            else:
                formatted_result['raw_bytes'] = "Raw bytes data not available"

            # 处理image信息
            if formatted.get('image'):
                image_info = self._extract_image_info(formatted['image'])
                formatted_result['image'] = image_info
            else:
                formatted_result['image'] = "Image data not available"

        return formatted_result

    def _extract_raw_bytes_info(self, raw_bytes_data: Dict) -> str:
        """从raw_bytes数据中提取关键信息"""
        try:
            info_parts = []

            # 文件头信息
            if raw_bytes_data.get('file_header'):
                header = raw_bytes_data['file_header']
                info_parts.append(f"File Header: {header}")

            # 节区信息
            if raw_bytes_data.get('sections'):
                sections = raw_bytes_data['sections']
                info_parts.append(f"Sections: {len(sections)} sections detected")
                for i, section in enumerate(sections[:3]):  # 只显示前3个节区
                    info_parts.append(f"  - Section {i+1}: {section.get('name', 'Unknown')}")

            # 导入表信息
            if raw_bytes_data.get('imports'):
                imports = raw_bytes_data['imports']
                info_parts.append(f"Imports: {len(imports)} imported functions")
                suspicious_imports = [imp for imp in imports if self._is_suspicious_import(imp)]
                if suspicious_imports:
                    info_parts.append("Suspicious Imports:")
                    for imp in suspicious_imports[:5]:
                        info_parts.append(f"  - {imp}")

            # 字符串特征
            if raw_bytes_data.get('strings'):
                strings = raw_bytes_data['strings']
                suspicious_strings = [s for s in strings if self._is_suspicious_string(s)]
                if suspicious_strings:
                    info_parts.append(f"Suspicious Strings: {len(suspicious_strings)} found")
                    for s in suspicious_strings[:5]:
                        info_parts.append(f"  - {s}")

            return "\n".join(info_parts) if info_parts else "Limited raw bytes information available"

        except Exception as e:
            logger.warning(f"Error extracting raw bytes info: {e}")
            return "Raw bytes information processing failed"

    def _extract_image_info(self, image_data: Dict) -> str:
        """从image数据中提取关键信息"""
        try:
            info_parts = []

            # 图像统计信息
            if image_data.get('statistics'):
                stats = image_data['statistics']
                info_parts.append("Image Statistics:")
                if 'entropy' in stats:
                    info_parts.append(f"  - Entropy: {stats['entropy']:.3f}")
                if 'mean_intensity' in stats:
                    info_parts.append(f"  - Mean Intensity: {stats['mean_intensity']:.3f}")

            # 纹理特征
            if image_data.get('texture_features'):
                texture = image_data['texture_features']
                info_parts.append("Texture Features:")
                if 'contrast' in texture:
                    info_parts.append(f"  - Contrast: {texture['contrast']:.3f}")
                if 'homogeneity' in texture:
                    info_parts.append(f"  - Homogeneity: {texture['homogeneity']:.3f}")

            # 颜色特征
            if image_data.get('color_features'):
                color = image_data['color_features']
                info_parts.append("Color Features:")
                if 'color_complexity' in color:
                    info_parts.append(f"  - Color Complexity: {color['color_complexity']:.3f}")

            return "\n".join(info_parts) if info_parts else "Limited image information available"

        except Exception as e:
            logger.warning(f"Error extracting image info: {e}")
            return "Image information processing failed"

    def _is_suspicious_import(self, import_name: str) -> bool:
        """判断导入函数是否可疑"""
        suspicious_keywords = [
            'crypt', 'encrypt', 'decrypt', 'virtualalloc', 'virtualprotect',
            'createremotethread', 'setwindowshook', 'regsetvalue',
            'getkeystate', 'shellexecute', 'winexec'
        ]
        import_lower = import_name.lower()
        return any(keyword in import_lower for keyword in suspicious_keywords)

    def _is_suspicious_string(self, string: str) -> bool:
        """判断字符串是否可疑"""
        suspicious_patterns = [
            'http://', 'https://', '.exe', '.dll', 'registry',
            'autostart', 'startup', 'password', 'keylogger'
        ]
        string_lower = string.lower()
        return any(pattern in string_lower for pattern in suspicious_patterns)

    def _call_llm(self, prompt: str, phase: str = "phase1") -> str:
        """调用LLM进行推理"""
        temperature=0.3 if phase == "phase1" else 0.1
        max_tokens= 800 if phase == "phase1" else 1200
        _, llm_response = llm_chat(0, [], prompt, self.llm_config["model"],t=temperature, l=max_tokens)
        return llm_response

    def _parse_phase1_response(self, response: str) -> Dict:
        """Parse phase 1 response"""
        parsed = {
            'preliminary_judgment': 'Unknown',
            'main_evidence': '',
            'confidence_assessment': 'Low',
            'reasoning_summary': ''
        }

        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if 'Preliminary Judgment' in line:
                    parsed['preliminary_judgment'] = line.replace('Preliminary Judgment:', '').strip()
                elif 'Main Evidence' in line:
                    parsed['main_evidence'] = line.replace('Main Evidence:', '').strip()
                elif 'Confidence Assessment' in line:
                    parsed['confidence_assessment'] = line.replace('Confidence Assessment:', '').strip()
                elif 'Reasoning Summary' in line:
                    parsed['reasoning_summary'] = line.replace('Reasoning Summary:', '').strip()

        except Exception as e:
            logger.warning(f"Failed to parse phase 1 response: {e}")

        return parsed

    def _parse_phase2_response(self, response: str) -> Dict:
        """Parse phase 2 response"""
        parsed = {
            'final_classification': 'Unknown',
            'key_evidence': '',
            'behavior_pattern': '',
            'detailed_reasoning': '',
            'final_confidence': 'Low'
        }

        try:
            lines = response.split('\n')
            current_section = 'more line'

            for line in lines:
                line = line.strip()
                if 'Final Classification' in line:
                    parsed['final_classification'] = 'Malicious' if 'Malicious' in line else 'Benign'
                    current_section = None
                elif 'Key Evidence' in line:
                    current_section = 'key_evidence'
                    parsed['key_evidence'] = line.replace('Key Evidence:', '').strip()
                elif  'Behavior Pattern'  in line:
                    current_section = 'behavior_pattern'
                    parsed['behavior_pattern'] = line.replace('Behavior Pattern:', '').strip()
                elif 'Detailed Reasoning:'  in line:
                    current_section = 'detailed_reasoning'
                    parsed['detailed_reasoning'] = line.replace('Detailed Reasoning:', '').strip()
                elif 'Final Confidence' in line:
                    parsed['final_confidence'] = line.replace('Final Confidence:', '').strip()
                    current_section = None

        except Exception as e:
            logger.warning(f"Failed to parse phase 2 response: {e}")

        return parsed
lock = threading.Lock()

class AVAgentPhase3Executor:
    """
    AV-Agent第三阶段执行器 - 整合特征工程和两阶段推理
    """

    def __init__(self, model_paths: Dict):
        self.model_paths = model_paths
        self.reasoning_engine = TwoPhaseReasoningEngine()

    def execute_av_agent_analysis(self, seq: int, conn) -> Dict:
        """
        执行完整的AV-Agent分析流程
        """
        logger.info(f"开始执行序列 {seq} 的AV-Agent分析")
        # 获取特征数据
        confidence_scores = get_classifier_confidence_scores(seq, conn)
        with lock:
            feature_results = extract_features_for_llm([seq],
                                                       joblib.load('drebin_model.pkl'),
                                                       joblib.load('tfidf_vectorizer.pkl'),
                                                       conn)

        if seq not in feature_results:
            raise ValueError(f"序列 {seq} 特征提取失败")

        feature_data = feature_results[seq]
        key_features = feature_data['key_features']

        # 获取formatted数据（包含raw_bytes和image信息）
        formatted = feature_data.get('formatted', {})

        # 第一阶段推理
        phase1_result = self.reasoning_engine.execute_phase1_reasoning(confidence_scores)

        # 第二阶段推理（传入formatted数据）
        phase2_result = self.reasoning_engine.execute_phase2_reasoning(phase1_result, key_features, formatted)

        # 整合结果
        final_result = {
            'seq': seq,
            'confidence_scores': confidence_scores,
            'key_features': key_features,
            'formatted_data': formatted,  # 包含原始数据信息
            'phase1_reasoning': phase1_result,
            'phase2_reasoning': phase2_result,
            'final_classification': phase2_result['parsed_result'].get('final_classification', 'unknown')
        }

        logger.info(f"序列 {seq} AV-Agent分析完成")
        return final_result
    def extract_av_agent_analysis(self, seq: int, conn) -> Dict:
        """
        执行完整的AV-Agent分析流程
        """
        logger.info(f"开始执行序列 {seq} 的AV-Agent分析")
        # 获取特征数据
        confidence_scores = get_classifier_confidence_scores(seq, conn)
        with lock:
            feature_results = extract_features_for_llm([seq],
                                                       joblib.load('drebin_model.pkl'),
                                                       joblib.load('tfidf_vectorizer.pkl'),
                                                       conn)

        if seq not in feature_results:
            raise ValueError(f"序列 {seq} 特征提取失败")

        feature_data = feature_results[seq]
        key_features = feature_data['key_features']

        # 获取formatted数据（包含raw_bytes和image信息）
        formatted = feature_data.get('formatted', {})

        # 直接加载test003已处理的结果
        cache_file = f"av_agent_output/seq_{str(seq)}_av_agent.json"
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_result = json.load(f)

        phase1_result = cached_result['phase1_reasoning']
        phase2_result = cached_result['phase2_reasoning']

        # 整合结果
        final_result = {
            'seq': seq,
            'confidence_scores': confidence_scores,
            'key_features': key_features,
            'formatted_data': formatted,  # 包含原始数据信息
            'phase1_reasoning': phase1_result,
            'phase2_reasoning': phase2_result,
            'final_classification': phase2_result['parsed_result'].get('final_classification', 'unknown')
        }

        return final_result

def run_av_agent_analysis(seqs: List[int], conn, output_dir: str = "av_agent_output"):
    """运行AV-Agent分析"""
    logger.info("开始AV-Agent分析流程")

    os.makedirs(output_dir, exist_ok=True)

    # 模型路径配置
    model_paths = {
        'raw_bytes': 'raw_bytes_model.pth',
        'drebin': 'drebin_model.pkl',
        'image': 'image_model.pth'
    }

    # 初始化执行器
    executorObj = AVAgentPhase3Executor(model_paths)

    max_workers = 3
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_seq = {
            executor.submit(process_single_seq,seq,output_dir, executorObj): seq
            for seq in seqs
        }

        # 处理完成的任务
        for future in as_completed(future_to_seq):
            seq, result = future.result()
            with lock:
                results[seq] = result

    # 保存所有结果
    if results:
        summary_file = os.path.join(output_dir, "all_av_agent_results.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"AV-Agent分析完成！结果保存至 {summary_file}")

    return results

def process_single_seq(seq,output_dir, executorObj):
    """处理单个序列的任务"""
    try:
        seq=str(seq)
        conn = get_connection()  # 根据需要实现
        logger.info(f"处理序列 {seq}")
        # 执行AV-Agent分析
        result = executorObj.execute_av_agent_analysis(seq, conn)

        # 保存单个结果
        output_file = os.path.join(output_dir, f"seq_{seq}_av_agent.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 打印摘要（注意：多线程下打印可能交错，如果需要顺序打印，可以加锁或稍后统一打印）
        #print_av_agent_summary(seq, result)
    finally:
        conn.close()
        return seq, result
def print_av_agent_summary(seq: int, result: Dict):
    """打印AV-Agent分析摘要"""
    print(f"\n=== 序列 {seq} AV-Agent分析结果 ===")
    print(f"最终分类: {result.get('final_classification', 'unknown')}")

    phase1_parsed = result.get('phase1_reasoning', {}).get('parsed_result', {})
    phase2_parsed = result.get('phase2_reasoning', {}).get('parsed_result', {})

    print(f"第一阶段判断: {phase1_parsed.get('preliminary_judgment', 'unknown')}")
    print(f"第二阶段置信度: {phase2_parsed.get('final_confidence', 'unknown')}")

    confidence_scores = result.get('confidence_scores', {})
    # print("分类器置信度:")
    # for feature_type, scores in confidence_scores.items():
    #     print(f"  - {scores['description']}: {scores['confidence']:.3f}")

def main(test_seqs = [52436]):
    """AV-Agent使用示例"""
    conn = get_connection()
    if conn is None:
        logger.error("无法连接到数据库")
        return


    try:
        # 运行AV-Agent分析
        results = run_av_agent_analysis(test_seqs, conn)

        # 打印总体统计
        if results:
            malicious_count = sum(1 for r in results.values()
                                  if r.get('final_classification') == 'Malicious')
            benign_count = sum(1 for r in results.values()
                               if r.get('final_classification') == 'Benign')

            print(f"\n=== 总体统计 ===")
            print(f"处理样本数: {len(results)}")
            print(f"恶意分类: {malicious_count}")
            print(f"良性分类: {benign_count}")
            print(f"未知分类: {len(results) - malicious_count - benign_count}")

    except Exception as e:
        logger.error(f"AV-Agent分析失败: {e}")
    finally:
        conn.close()
        logger.info("数据库连接已关闭")

if __name__ == "__main__":
    main()