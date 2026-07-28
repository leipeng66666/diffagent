#!/usr/bin/env python3
"""
表格数据可视化问答AI Agent - 使用示例
"""
import pandas as pd
import numpy as np
from table_agent import TableAgent
from loguru import logger

def create_sample_data():
    """创建示例数据"""
    logger.info("创建示例数据")
    
    # 创建示例数据集
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Material': np.random.choice(['CO2', 'H2O', 'N2', 'O2', 'CH4'], n_samples),
        'Temperature': np.random.normal(300, 50, n_samples),
        'Pressure': np.random.normal(1.0, 0.3, n_samples),
        'Concentration': np.random.exponential(2.0, n_samples),
        'Time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Quality': np.random.choice(['High', 'Medium', 'Low'], n_samples, p=[0.3, 0.5, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些异常值
    df.loc[df['Temperature'] > 400, 'Temperature'] = np.random.normal(450, 20, len(df[df['Temperature'] > 400]))
    df.loc[df['Pressure'] > 1.5, 'Pressure'] = np.random.normal(1.8, 0.1, len(df[df['Pressure'] > 1.5]))
    
    # 保存为CSV文件
    df.to_csv('sample_data.csv', index=False)
    logger.info(f"示例数据已保存: sample_data.csv (形状: {df.shape})")
    
    return df

def demo_basic_usage():
    """演示基本使用"""
    logger.info("=== 基本使用演示 ===")
    
    # 创建AI Agent
    agent = TableAgent()
    
    # 创建示例数据
    sample_data = create_sample_data()
    
    # 加载数据
    result = agent.load_table('sample_data.csv')
    if not result['success']:
        logger.error(f"加载数据失败: {result['message']}")
        return
    
    logger.info("数据加载成功")
    
    # 获取数据预览
    preview = agent.get_data_preview(5)
    logger.info(f"数据预览: {preview['shape']}")
    
    # 获取列信息
    column_info = agent.get_column_info()
    logger.info(f"列信息: {list(column_info.keys())}")
    
    return agent

def demo_query_processing(agent):
    """演示查询处理"""
    logger.info("=== 查询处理演示 ===")
    
    queries = [
        "分析温度分布",
        "比较不同材料的性能",
        "显示二氧化碳浓度大于2的数据",
        "绘制温度和压力的散点图",
        "分析质量等级分布"
    ]
    
    for query in queries:
        logger.info(f"处理查询: {query}")
        
        result = agent.process_query(query)
        
        if result['success']:
            logger.info(f"回答: {result['response']['answer'][:200]}...")
            
            if result['visualizations']:
                logger.info(f"生成了 {len(result['visualizations'])} 个可视化图表")
            
            if result['insights']:
                logger.info(f"洞察: {result['insights']['summary']}")
        else:
            logger.error(f"查询失败: {result['message']}")

def demo_visualization(agent):
    """演示可视化功能"""
    logger.info("=== 可视化演示 ===")
    
    # 获取可视化建议
    suggestions = agent.visualization_engine.get_visualization_suggestions(agent.current_data)
    logger.info(f"可视化建议: {len(suggestions)} 个")
    
    for suggestion in suggestions:
        logger.info(f"- {suggestion['type']}: {suggestion['description']}")
    
    # 创建特定图表
    try:
        # 创建直方图
        hist_result = agent.visualization_engine.create_histogram(
            agent.current_data, 'Temperature', title="温度分布图"
        )
        if 'error' not in hist_result:
            logger.info("温度分布图创建成功")
        
        # 创建散点图
        scatter_result = agent.visualization_engine.create_scatter_plot(
            agent.current_data, 'Temperature', 'Pressure', title="温度-压力散点图"
        )
        if 'error' not in scatter_result:
            logger.info("温度-压力散点图创建成功")
        
        # 创建柱状图
        bar_result = agent.visualization_engine.create_bar_chart(
            agent.current_data, 'Material', 'Temperature', title="材料温度对比"
        )
        if 'error' not in bar_result:
            logger.info("材料温度对比图创建成功")
            
    except Exception as e:
        logger.error(f"可视化创建失败: {e}")

def demo_advanced_features(agent):
    """演示高级功能"""
    logger.info("=== 高级功能演示 ===")
    
    # 添加自定义同义词映射
    agent.add_custom_synonym_mapping(
        key="反应温度",
        synonyms=["reaction_temperature", "react_temp", "T_reaction"]
    )
    logger.info("添加自定义同义词映射")
    
    # 导出分析报告
    try:
        report_result = agent.export_analysis_report(
            query="分析温度分布和材料性能",
            output_path="analysis_report.json"
        )
        if report_result['success']:
            logger.info(f"分析报告已导出: {report_result['output_path']}")
        else:
            logger.error(f"报告导出失败: {report_result['error']}")
    except Exception as e:
        logger.error(f"报告导出失败: {e}")
    
    # 获取系统状态
    status = agent.get_system_status()
    logger.info(f"系统状态: {status}")

def demo_batch_analysis(agent):
    """演示批量分析"""
    logger.info("=== 批量分析演示 ===")
    
    batch_queries = [
        "分析温度分布特征",
        "比较不同材料的平均压力",
        "识别异常值",
        "分析时间序列趋势",
        "计算变量相关性"
    ]
    
    results = []
    for query in batch_queries:
        logger.info(f"批量处理: {query}")
        result = agent.process_query(query)
        results.append({
            'query': query,
            'success': result['success'],
            'answer_length': len(result['response']['answer']) if result['success'] else 0
        })
    
    # 统计结果
    successful = sum(1 for r in results if r['success'])
    logger.info(f"批量分析完成: {successful}/{len(results)} 个查询成功")

def main():
    """主函数"""
    logger.info("开始表格数据AI Agent演示")
    
    try:
        # 基本使用
        agent = demo_basic_usage()
        if not agent:
            return
        
        # 查询处理
        demo_query_processing(agent)
        
        # 可视化功能
        demo_visualization(agent)
        
        # 高级功能
        demo_advanced_features(agent)
        
        # 批量分析
        demo_batch_analysis(agent)
        
        logger.info("演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()



