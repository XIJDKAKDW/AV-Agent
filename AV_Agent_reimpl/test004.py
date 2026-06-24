# coding=utf-8
import os
import json
import logging
import platform
import sys
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score
system = platform.system()
def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/combine_compare_tool_method")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/ganerate_pic_graph")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test002")

from combine_compare_tool_method import get_connection
from test003 import AVAgentPhase3Executor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class AVAgentSystem:
    """
    AV-Agent完整系统 - 整合所有模块实现论文描述的完整流程
    """

    def __init__(self, model_paths: Dict):
        self.model_paths = model_paths
        self.performance_metrics = {}
        self.interpretability_records = {}

    def run_complete_av_agent(self, seqs: List[int], conn, output_dir: str = "av_agent_final") -> Dict:
        """
        运行完整的AV-Agent系统
        对应论文中的完整工作流程
        """
        logger.info("开始运行完整AV-Agent系统")

        os.makedirs(output_dir, exist_ok=True)

        results = {}
        performance_data = []

        for seq in seqs:
            try:
                logger.info(f"处理序列 {seq}")

                # 执行完整的AV-Agent分析流程
                av_agent_result = self._execute_av_agent_pipeline(seq, conn)

                if av_agent_result:
                    results[seq] = av_agent_result

                    # 收集性能数据
                    performance_entry = self._collect_performance_metrics(seq, av_agent_result, conn)
                    performance_data.append(performance_entry)

                    # 保存可解释性记录
                    self._save_interpretability_record(seq, av_agent_result, output_dir)

                    # 打印实时结果
                    self._print_realtime_result(seq, av_agent_result)

            except Exception as e:
                logger.error(f"处理序列 {seq} 失败: {e}")
                continue

        # 生成性能报告
        if performance_data:
            self._generate_performance_report(performance_data, output_dir)

        # 保存最终结果
        if results:
            self._save_final_results(results, output_dir)

        logger.info("完整AV-Agent系统执行完成")
        return results

    def _execute_av_agent_pipeline(self, seq: int, conn) -> Dict:
        """
        执行AV-Agent完整流水线
        对应论文中的完整算法流程
        """
        try:
            # 初始化执行器
            executor = AVAgentPhase3Executor(self.model_paths)

            # 执行分析
            result = executor.extract_av_agent_analysis(seq, conn)

            # 添加时间戳和元数据
            result['timestamp'] = datetime.now().isoformat()
            result['av_agent_version'] = '1.0'
            result['pipeline_steps'] = [
                'Hierarchical Feature Construction',
                'Character Feature Extraction',
                'Two-Phase Reasoning',
                'Final Classification Decision'
            ]

            return result

        except Exception as e:
            logger.error(f"AV-Agent流水线执行失败: {e}")
            return None

    def _collect_performance_metrics(self, seq: int, result: Dict, conn) -> Dict:
        """收集性能指标"""
        try:
            # 获取真实标签
            true_label = self._get_true_label(seq, conn)

            # 获取预测结果
            pred_classification = result.get('final_classification', 'Unknown')
            pred_label = 1 if pred_classification == 'Malicious' else 0 if pred_classification == 'Benign' else -1

            # 计算置信度统计
            confidence_scores = result.get('confidence_scores', {})
            conf_values = [scores.get('confidence', 0) for scores in confidence_scores.values()]
            avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0

            # 推理质量评估
            reasoning_quality = self._evaluate_reasoning_quality(result)

            performance_entry = {
                'seq': seq,
                'true_label': true_label,
                'predicted_label': pred_label,
                'final_classification': pred_classification,
                'avg_confidence': avg_confidence,
                'reasoning_quality': reasoning_quality,
                'timestamp': datetime.now().isoformat()
            }

            return performance_entry

        except Exception as e:
            logger.error(f"性能指标收集失败: {e}")
            return {}

    def _evaluate_reasoning_quality(self, result: Dict) -> Dict:
        """评估推理质量"""
        quality = {
            'consistency': 'Unknown',
            'reasoning_depth': 'Shallow',
            'evidence_utilization': 'Low',
            'overall_quality': 'Low'
        }

        try:
            phase1_parsed = result.get('phase1_reasoning', {}).get('parsed_result', {})
            phase2_parsed = result.get('phase2_reasoning', {}).get('parsed_result', {})

            # 评估一致性
            phase1_judgment = phase1_parsed.get('preliminary_judgment', '')
            phase2_judgment = phase2_parsed.get('final_classification', '')

            if phase1_judgment and phase2_judgment:
                if phase1_judgment in phase2_judgment or phase2_judgment in phase1_judgment:
                    quality['consistency'] = 'High'
                else:
                    quality['consistency'] = 'Low'

            # 评估推理深度
            reasoning_text = phase2_parsed.get('detailed_reasoning', '')
            if len(reasoning_text) > 200:
                quality['reasoning_depth'] = 'Deep'
            elif len(reasoning_text) > 100:
                quality['reasoning_depth'] = 'Medium'
            else:
                quality['reasoning_depth'] = 'Shallow'

            # 评估证据利用
            key_evidence = phase2_parsed.get('key_evidence', '')
            feature_count = sum(1 for feature_type in ['raw_bytes', 'drebin', 'image']
                                if feature_type in key_evidence)

            if feature_count >= 2:
                quality['evidence_utilization'] = 'High'
            elif feature_count >= 1:
                quality['evidence_utilization'] = 'Medium'
            else:
                quality['evidence_utilization'] = 'Low'

            # 总体质量评估
            if (quality['consistency'] == 'High' and
                    quality['reasoning_depth'] in ['Deep', 'Medium'] and
                    quality['evidence_utilization'] in ['High', 'Medium']):
                quality['overall_quality'] = 'High'
            elif (quality['consistency'] == 'High' or
                  quality['reasoning_depth'] == 'Medium'):
                quality['overall_quality'] = 'Medium'
            else:
                quality['overall_quality'] = 'Low'

        except Exception as e:
            logger.warning(f"推理质量评估失败: {e}")

        return quality

    def _get_true_label(self, seq: int, conn) -> int:
        """获取真实标签"""
        try:
            with conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"获取真实标签失败: {e}")
        return -1

    def _save_interpretability_record(self, seq: int, result: Dict, output_dir: str):
        """保存可解释性记录"""
        try:
            interpretability_data = {
                'seq': seq,
                'confidence_scores': result.get('confidence_scores', {}),
                'key_features': result.get('key_features', {}),
                'phase1_reasoning': result.get('phase1_reasoning', {}),
                'phase2_reasoning': result.get('phase2_reasoning', {}),
                'final_classification': result.get('final_classification', ''),
                'timestamp': datetime.now().isoformat()
            }

            record_file = os.path.join(output_dir, f"seq_{seq}_interpretability.json")
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(interpretability_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"可解释性记录保存失败: {e}")

    def _generate_performance_report(self, performance_data: List[Dict], output_dir: str):
        """生成性能报告"""
        try:
            df = pd.DataFrame(performance_data)

            # 计算基本指标
            total_samples = len(df)
            correct_predictions = len(df[df['true_label'] == df['predicted_label']])
            # accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            y_true = df['true_label']
            y_pred = df['predicted_label']
            # 使用平衡准确率代替普通准确率
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

            # 恶意样本检测指标
            malicious_samples = df[df['true_label'] == 1]
            true_positives = len(malicious_samples[malicious_samples['predicted_label'] == 1])
            false_negatives = len(malicious_samples[malicious_samples['predicted_label'] == 0])

            recall = true_positives / len(malicious_samples) if len(malicious_samples) > 0 else 0

            # 良性样本检测指标
            benign_samples = df[df['true_label'] == 0]
            true_negatives = len(benign_samples[benign_samples['predicted_label'] == 0])
            false_positives = len(benign_samples[benign_samples['predicted_label'] == 1])

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

            # 推理质量统计
            reasoning_qualities = [entry['reasoning_quality']['overall_quality'] for entry in performance_data]
            quality_distribution = {
                'High': reasoning_qualities.count('High'),
                'Medium': reasoning_qualities.count('Medium'),
                'Low': reasoning_qualities.count('Low')
            }

            # 生成报告
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': total_samples,
                'balanced_accuracy': balanced_accuracy,
                'recall': recall,
                'precision': precision,
                'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
                'avg_confidence': df['avg_confidence'].mean(),
                'reasoning_quality_distribution': quality_distribution,
                'detailed_metrics': {
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'true_negatives': true_negatives,
                    'false_negatives': false_negatives
                }
            }

            # 保存报告
            report_file = os.path.join(output_dir, "performance_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # 生成CSV格式的详细数据
            csv_file = os.path.join(output_dir, "performance_details.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8')

            logger.info(f"性能报告生成完成: {report_file}")
            print(f"AV-Agent性能: {report}",flush=True)

        except Exception as e:
            logger.error(f"性能报告生成失败: {e}")

    def _save_final_results(self, results: Dict, output_dir: str):
        """保存最终结果"""
        try:
            # 保存完整结果
            final_file = os.path.join(output_dir, "av_agent_final_results.json")
            with open(final_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            # 生成摘要报告
            summary = self._generate_summary_report(results)
            summary_file = os.path.join(output_dir, "summary_report.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"最终结果保存完成: {final_file}")

        except Exception as e:
            logger.error(f"最终结果保存失败: {e}")

    def _generate_summary_report(self, results: Dict) -> Dict:
        """生成摘要报告"""
        summary = {
            'total_processed': len(results),
            'classification_distribution': {},
            'avg_confidence_by_type': {},
            'reasoning_quality_summary': {}
        }

        # 分类分布
        classifications = [result.get('final_classification', 'Unknown') for result in results.values()]
        summary['classification_distribution'] = {
            'Malicious': classifications.count('Malicious'),
            'Benign': classifications.count('Benign'),
            'Unknown': classifications.count('Unknown')
        }

        # 平均置信度
        confidences_by_type = {'raw_bytes': [], 'drebin': [], 'image': []}
        for result in results.values():
            confidence_scores = result.get('confidence_scores', {})
            for feature_type, scores in confidence_scores.items():
                if feature_type in confidences_by_type:
                    confidences_by_type[feature_type].append(scores.get('confidence', 0))

        for feature_type, conf_list in confidences_by_type.items():
            summary['avg_confidence_by_type'][feature_type] = sum(conf_list) / len(conf_list) if conf_list else 0

        return summary

    def _print_realtime_result(self, seq: int, result: Dict):
        """打印实时结果"""
        final_classification = result.get('final_classification', 'Unknown')
        phase1_judgment = result.get('phase1_reasoning', {}).get('parsed_result', {}).get('preliminary_judgment', 'Unknown')
        phase2_confidence = result.get('phase2_reasoning', {}).get('parsed_result', {}).get('final_confidence', 'Unknown')
        print(f"[AV-Agent] Sequence {seq}: Final Classification={final_classification}, "
              f"Phase1 Judgment={phase1_judgment}, Phase2 Confidence={phase2_confidence}")

def main(test_seqs = [27141,26506,92417,26184,109850,26705,14548,27342,12659,25750]):
    """AV-Agent完整系统使用示例"""
    conn = get_connection()
    if conn is None:
        logger.error("无法连接到数据库")
        return

    # 模型路径配置
    model_paths = {
        'raw_bytes': 'raw_bytes_model.pth',
        'drebin': 'drebin_model.pkl',
        'image': 'image_model.pth'
    }

    # 测试序列


    try:
        # 初始化AV-Agent系统
        av_agent_system = AVAgentSystem(model_paths)

        # 运行完整系统
        results = av_agent_system.run_complete_av_agent(test_seqs, conn)

        print(f"\n=== AV-Agent系统执行完成 ===")
        print(f"处理样本数: {len(results)}")

        # 打印最终统计
        malicious_count = sum(1 for r in results.values() if r.get('final_classification') == 'Malicious')
        benign_count = sum(1 for r in results.values() if r.get('final_classification') == 'Benign')

        print(f"恶意分类: {malicious_count}")
        print(f"良性分类: {benign_count}")
        print(f"详细报告请查看输出目录")

    except Exception as e:
        logger.error(f"AV-Agent系统执行失败: {e}")
    finally:
        conn.close()
        logger.info("数据库连接已关闭")

if __name__ == "__main__":
    main([16983])