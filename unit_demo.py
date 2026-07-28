#!/usr/bin/env python3
"""
单位识别功能演示
"""
import pandas as pd
import numpy as np
from core.unit_recognizer import UnitRecognizer
from core.semantic_parser import SemanticParser
from loguru import logger

def create_sample_data_with_units():
    """创建包含单位的示例数据"""
    logger.info("创建包含单位的示例数据")
    
    # 创建包含各种单位的示例数据
    data = {
        'Temperature': ['25°C', '30°C', '35°C', '40°C', '45°C'],
        'Pressure': ['1.0 bar', '1.2 bar', '1.5 bar', '2.0 bar', '2.5 bar'],
        'Concentration': ['0.5 M', '1.0 M', '1.5 M', '2.0 M', '2.5 M'],
        'Volume': ['100 mL', '200 mL', '300 mL', '400 mL', '500 mL'],
        'Mass': ['10 g', '20 g', '30 g', '40 g', '50 g'],
        'Time': ['1 h', '2 h', '3 h', '4 h', '5 h'],
        'Length': ['10 cm', '20 cm', '30 cm', '40 cm', '50 cm']
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_data_with_units.csv', index=False)
    logger.info(f"示例数据已保存: sample_data_with_units.csv (形状: {df.shape})")
    
    return df

def demo_unit_recognition():
    """演示单位识别功能"""
    logger.info("=== 单位识别功能演示 ===")
    
    # 创建单位识别器
    unit_recognizer = UnitRecognizer()
    
    # 测试查询中的单位识别
    test_queries = [
        "分析温度大于30°C的数据",
        "显示压力在1.5 bar到2.0 bar之间的记录",
        "比较浓度为1.0 M和2.0 M的差异",
        "绘制体积为200 mL到400 mL的分布图",
        "分析质量大于25 g的样本",
        "显示时间在2 h到4 h范围内的数据",
        "比较长度为10 cm和50 cm的差异"
    ]
    
    for query in test_queries:
        logger.info(f"查询: {query}")
        
        # 提取单位信息
        units = unit_recognizer.extract_units_from_text(query)
        
        if units:
            logger.info(f"识别到的单位:")
            for unit in units:
                logger.info(f"  - 值: {unit.value}, 单位: {unit.unit}, 置信度: {unit.confidence:.2f}")
        else:
            logger.info("  未识别到单位")
        
        print()

def demo_unit_conversion():
    """演示单位转换功能"""
    logger.info("=== 单位转换功能演示 ===")
    
    unit_recognizer = UnitRecognizer()
    
    # 测试单位转换
    conversion_tests = [
        (100, '°F', '°C', 'temperature'),
        (1.0, 'bar', 'Pa', 'pressure'),
        (1000, 'mg/L', 'g/L', 'concentration'),
        (1.0, 'L', 'mL', 'volume'),
        (1000, 'g', 'kg', 'mass')
    ]
    
    for value, from_unit, to_unit, category in conversion_tests:
        converted = unit_recognizer.convert_units(value, from_unit, to_unit, category)
        if converted is not None:
            logger.info(f"{value} {from_unit} = {converted:.2f} {to_unit}")
        else:
            logger.info(f"无法转换 {value} {from_unit} 到 {to_unit}")

def demo_data_unit_analysis():
    """演示数据单位分析"""
    logger.info("=== 数据单位分析演示 ===")
    
    # 创建包含单位的示例数据
    sample_data = create_sample_data_with_units()
    
    unit_recognizer = UnitRecognizer()
    
    # 分析数据中的单位
    all_text = []
    for col in sample_data.columns:
        all_text.extend(sample_data[col].astype(str).tolist())
    
    unit_analysis = unit_recognizer.analyze_data_units(all_text)
    
    logger.info("单位分析结果:")
    logger.info(f"总共找到 {unit_analysis['total_units_found']} 个单位")
    logger.info(f"唯一单位数量: {unit_analysis['unique_units']}")
    
    logger.info("\n单位统计:")
    for unit, stats in unit_analysis['unit_statistics'].items():
        logger.info(f"  {unit}: 出现 {stats['count']} 次")
        logger.info(f"    范围: {stats['min']:.2f} - {stats['max']:.2f}")
        logger.info(f"    平均值: {stats['mean']:.2f}")
        logger.info(f"    标准差: {stats['std']:.2f}")
    
    logger.info("\n建议:")
    for recommendation in unit_analysis['recommendations']:
        logger.info(f"  - {recommendation}")

def demo_semantic_parser_with_units():
    """演示带单位识别的语义解析"""
    logger.info("=== 带单位识别的语义解析演示 ===")
    
    semantic_parser = SemanticParser()
    
    test_queries = [
        "分析温度大于30°C的数据分布",
        "显示压力在1.5 bar到2.0 bar之间的记录",
        "比较浓度为1.0 M和2.0 M的性能差异",
        "绘制体积为200 mL到400 mL的散点图",
        "分析质量大于25 g的样本特征"
    ]
    
    for query in test_queries:
        logger.info(f"查询: {query}")
        
        # 解析查询
        result = semantic_parser.parse_query(query)
        
        logger.info("解析结果:")
        logger.info(f"  意图: {result['intent']}")
        logger.info(f"  实体数量: {len(result['entities'])}")
        logger.info(f"  条件数量: {len(result['conditions'])}")
        logger.info(f"  单位数量: {len(result['units'])}")
        logger.info(f"  置信度: {result['confidence']:.2f}")
        
        # 显示识别的单位
        if result['units']:
            logger.info("  识别的单位:")
            for unit in result['units']:
                logger.info(f"    - {unit.value} {unit.unit} (置信度: {unit.confidence:.2f})")
        
        print()

def demo_unit_normalization():
    """演示单位标准化"""
    logger.info("=== 单位标准化演示 ===")
    
    unit_recognizer = UnitRecognizer()
    
    # 创建包含不同单位的测试数据
    test_units = [
        unit_recognizer.UnitInfo(25, '°C', '25°C', 0.9),
        unit_recognizer.UnitInfo(77, '°F', '77°F', 0.9),
        unit_recognizer.UnitInfo(298, 'K', '298K', 0.9),
        unit_recognizer.UnitInfo(1.0, 'bar', '1.0 bar', 0.9),
        unit_recognizer.UnitInfo(100000, 'Pa', '100000 Pa', 0.9),
        unit_recognizer.UnitInfo(1.0, 'M', '1.0 M', 0.9),
        unit_recognizer.UnitInfo(1000, 'mg/L', '1000 mg/L', 0.9)
    ]
    
    # 标准化温度单位
    normalized_temp = unit_recognizer.normalize_units(test_units[:3], 'temperature')
    logger.info("温度单位标准化:")
    for unit in normalized_temp:
        logger.info(f"  {unit.value} {unit.unit}")
    
    # 标准化压力单位
    normalized_pressure = unit_recognizer.normalize_units(test_units[3:5], 'pressure')
    logger.info("压力单位标准化:")
    for unit in normalized_pressure:
        logger.info(f"  {unit.value} {unit.unit}")
    
    # 标准化浓度单位
    normalized_conc = unit_recognizer.normalize_units(test_units[5:7], 'concentration')
    logger.info("浓度单位标准化:")
    for unit in normalized_conc:
        logger.info(f"  {unit.value} {unit.unit}")

def main():
    """主函数"""
    logger.info("开始单位识别功能演示")
    
    try:
        # 单位识别演示
        demo_unit_recognition()
        
        # 单位转换演示
        demo_unit_conversion()
        
        # 数据单位分析演示
        demo_data_unit_analysis()
        
        # 语义解析与单位识别演示
        demo_semantic_parser_with_units()
        
        # 单位标准化演示
        demo_unit_normalization()
        
        logger.info("单位识别功能演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()



