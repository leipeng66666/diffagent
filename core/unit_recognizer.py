"""
Unit Recognition Module - Identify and process unit information in data
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger

@dataclass
class UnitInfo:
    """Unit Information"""
    value: float
    unit: str
    original_text: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'value': self.value,
            'unit': self.unit,
            'original_text': self.original_text,
            'confidence': self.confidence
        }

class UnitRecognizer:
    """Unit Recognizer"""
    
    def __init__(self):
        """Initialize unit recognizer"""
        # Unit pattern definitions
        self.unit_patterns = {
            # Temperature units
            'temperature': {
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*°?[Cc]',  # Celsius
                    r'(\d+(?:\.\d+)?)\s*°?[Ff]',  # Fahrenheit
                    r'(\d+(?:\.\d+)?)\s*[Kk]',   # Kelvin
                    r'(\d+(?:\.\d+)?)\s*度',     # Chinese degree
                ],
                'units': ['°C', '°F', 'K', '度', 'C', 'F']
            },
            
            # Pressure units
            'pressure': {
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*[Pp][Aa]',      # Pascal
                    r'(\d+(?:\.\d+)?)\s*[Kk][Pp][Aa]',  # Kilopascal
                    r'(\d+(?:\.\d+)?)\s*[Mm][Pp][Aa]',  # Megapascal
                    r'(\d+(?:\.\d+)?)\s*[Bb][Aa][Rr]',  # Bar
                    r'(\d+(?:\.\d+)?)\s*[Aa][Tt][Mm]',  # Atmosphere
                    r'(\d+(?:\.\d+)?)\s*[Mm][Mm][Hh][Gg]', # mmHg
                    r'(\d+(?:\.\d+)?)\s*[Pp][Ss][Ii]',  # PSI
                ],
                'units': ['Pa', 'kPa', 'MPa', 'bar', 'atm', 'mmHg', 'psi']
            },
            
            # Concentration units
            'concentration': {
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*[Mm]',          # Molar
                    r'(\d+(?:\.\d+)?)\s*[Mm][Oo][Ll]',  # Mole
                    r'(\d+(?:\.\d+)?)\s*[Gg]/[Ll]',      # g/L
                    r'(\d+(?:\.\d+)?)\s*[Mm][Gg]/[Ll]',  # mg/L
                    r'(\d+(?:\.\d+)?)\s*[Pp][Pp][Mm]',   # ppm
                    r'(\d+(?:\.\d+)?)\s*[Pp][Pp][Bb]',   # ppb
                    r'(\d+(?:\.\d+)?)\s*%',              # Percent
                    r'(\d+(?:\.\d+)?)\s*浓度',           # Chinese concentration
                ],
                'units': ['M', 'mol', 'g/L', 'mg/L', 'ppm', 'ppb', '%', '浓度']
            },
            
            # Mass units
            'mass': {
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*[Gg]',          # Gram
                    r'(\d+(?:\.\d+)?)\s*[Kk][Gg]',      # Kilogram
                    r'(\d+(?:\.\d+)?)\s*[Mm][Gg]',       # Milligram
                    r'(\d+(?:\.\d+)?)\s*[Ll][Bb]',       # Pound
                    r'(\d+(?:\.\d+)?)\s*[Oo][Zz]',       # Ounce
                    r'(\d+(?:\.\d+)?)\s*[Tt]',           # Ton
                    r'(\d+(?:\.\d+)?)\s*克',             # Chinese gram
                    r'(\d+(?:\.\d+)?)\s*千克',           # Chinese kilogram
                ],
                'units': ['g', 'kg', 'mg', 'lb', 'oz', 't', '克', '千克']
            },
            
            # Volume units
            'volume': {
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*[Mm][Ll]',      # Milliliter
                    r'(\d+(?:\.\d+)?)\s*[Ll]',           # Liter
                    r'(\d+(?:\.\d+)?)\s*[Cc][Mm]³',      # Cubic centimeter
                    r'(\d+(?:\.\d+)?)\s*[Mm]³',          # Cubic meter
                    r'(\d+(?:\.\d+)?)\s*[Gg][Aa][Ll]',   # Gallon
                    r'(\d+(?:\.\d+)?)\s*毫升',           # Chinese milliliter
                    r'(\d+(?:\.\d+)?)\s*升',             # Chinese liter
                ],
                'units': ['mL', 'L', 'cm³', 'm³', 'gal', '毫升', '升']
            },
            
            # Length units
            'length': {
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*[Mm][Mm]',      # Millimeter
                    r'(\d+(?:\.\d+)?)\s*[Cc][Mm]',       # Centimeter
                    r'(\d+(?:\.\d+)?)\s*[Mm]',           # Meter
                    r'(\d+(?:\.\d+)?)\s*[Kk][Mm]',       # Kilometer
                    r'(\d+(?:\.\d+)?)\s*[Ii][Nn]',       # Inch
                    r'(\d+(?:\.\d+)?)\s*[Ff][Tt]',       # Foot
                    r'(\d+(?:\.\d+)?)\s*毫米',           # Chinese millimeter
                    r'(\d+(?:\.\d+)?)\s*厘米',           # Chinese centimeter
                    r'(\d+(?:\.\d+)?)\s*米',             # Chinese meter
                ],
                'units': ['mm', 'cm', 'm', 'km', 'in', 'ft', '毫米', '厘米', '米']
            },
            
            # Time units
            'time': {
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*[Ss]',          # Second
                    r'(\d+(?:\.\d+)?)\s*[Mm][Ii][Nn]',   # Minute
                    r'(\d+(?:\.\d+)?)\s*[Hh]',          # Hour
                    r'(\d+(?:\.\d+)?)\s*[Dd]',          # Day
                    r'(\d+(?:\.\d+)?)\s*[Ww]',          # Week
                    r'(\d+(?:\.\d+)?)\s*[Mm][Oo][Nn]',  # Month
                    r'(\d+(?:\.\d+)?)\s*[Yy]',          # Year
                    r'(\d+(?:\.\d+)?)\s*秒',            # Chinese second
                    r'(\d+(?:\.\d+)?)\s*分钟',          # Chinese minute
                    r'(\d+(?:\.\d+)?)\s*小时',          # Chinese hour
                ],
                'units': ['s', 'min', 'h', 'd', 'w', 'mon', 'y', '秒', '分钟', '小时']
            }
        }
        
        # Unit conversion factors
        self.conversion_factors = {
            'temperature': {
                '°C': 1.0,
                '°F': lambda x: (x - 32) * 5/9,  # Fahrenheit to Celsius
                'K': lambda x: x - 273.15,       # Kelvin to Celsius
            },
            'pressure': {
                'Pa': 1.0,
                'kPa': 1000.0,
                'MPa': 1000000.0,
                'bar': 100000.0,
                'atm': 101325.0,
                'mmHg': 133.322,
                'psi': 6894.76,
            },
            'concentration': {
                'M': 1.0,
                'mol': 1.0,
                'g/L': 1.0,
                'mg/L': 0.001,
                'ppm': 0.001,
                'ppb': 0.000001,
                '%': 0.01,
            }
        }
    
    def extract_units_from_text(self, text: str) -> List[UnitInfo]:
        """Extract unit information from text"""
        logger.info(f"Extracting units from text: {text}")
        
        units_found = []
        
        for category, config in self.unit_patterns.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = float(match.group(1))
                    unit = self._extract_unit_from_match(match.group(0), config['units'])
                    
                    if unit:
                        unit_info = UnitInfo(
                            value=value,
                            unit=unit,
                            original_text=match.group(0),
                            confidence=self._calculate_unit_confidence(match.group(0), unit)
                        )
                        units_found.append(unit_info)
        
        # Deduplicate
        unique_units = self._deduplicate_units(units_found)
        
        logger.info(f"Extracted {len(unique_units)} unit information")
        return unique_units
    
    def _extract_unit_from_match(self, match_text: str, possible_units: List[str]) -> Optional[str]:
        """Extract unit from match text"""
        match_text = match_text.strip()
        
        for unit in possible_units:
            if unit.lower() in match_text.lower():
                return unit
        
        return None
    
    def _calculate_unit_confidence(self, match_text: str, unit: str) -> float:
        """Calculate unit recognition confidence"""
        # Based on match length and unit clarity
        base_confidence = 0.8
        
        # Increase confidence if unit explicitly appears in text
        if unit.lower() in match_text.lower():
            base_confidence += 0.1
        
        # Increase confidence if contains special symbols (like °, /, etc.)
        if any(char in match_text for char in ['°', '/', '³', '²']):
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _deduplicate_units(self, units: List[UnitInfo]) -> List[UnitInfo]:
        """Deduplicate unit information"""
        seen = set()
        unique_units = []
        
        for unit in units:
            key = (unit.value, unit.unit)
            if key not in seen:
                seen.add(key)
                unique_units.append(unit)
        
        return unique_units
    
    def convert_units(self, value: float, from_unit: str, to_unit: str, 
                    category: str = None) -> Optional[float]:
        """Convert units"""
        if category and category in self.conversion_factors:
            factors = self.conversion_factors[category]
            
            if from_unit in factors and to_unit in factors:
                if callable(factors[from_unit]):
                    # Use conversion function
                    base_value = factors[from_unit](value)
                else:
                    # Use conversion factor
                    base_value = value / factors[from_unit]
                
                if callable(factors[to_unit]):
                    return factors[to_unit](base_value)
                else:
                    return base_value * factors[to_unit]
        
        return None
    
    def normalize_units(self, units: List[UnitInfo], target_category: str) -> List[UnitInfo]:
        """Normalize units to target category"""
        if target_category not in self.conversion_factors:
            return units
        
        normalized_units = []
        target_unit = list(self.conversion_factors[target_category].keys())[0]
        
        for unit in units:
            # Find the corresponding category
            unit_category = self._find_unit_category(unit.unit)
            
            if unit_category == target_category:
                # Convert to target unit
                converted_value = self.convert_units(
                    unit.value, unit.unit, target_unit, unit_category
                )
                
                if converted_value is not None:
                    normalized_unit = UnitInfo(
                        value=converted_value,
                        unit=target_unit,
                        original_text=unit.original_text,
                        confidence=unit.confidence
                    )
                    normalized_units.append(normalized_unit)
            else:
                # Keep original unit
                normalized_units.append(unit)
        
        return normalized_units
    
    def _find_unit_category(self, unit: str) -> Optional[str]:
        """Find the category a unit belongs to"""
        for category, config in self.unit_patterns.items():
            if unit in config['units']:
                return category
        return None
    
    def analyze_data_units(self, data: List[str]) -> Dict[str, Any]:
        """Analyze unit information in data"""
        logger.info("Analyzing data unit information")
        
        all_units = []
        unit_statistics = {}
        
        for text in data:
            if isinstance(text, str):
                units = self.extract_units_from_text(text)
                all_units.extend(units)
        
        # Statistics on unit usage
        unit_counts = {}
        for unit in all_units:
            unit_key = unit.unit
            if unit_key not in unit_counts:
                unit_counts[unit_key] = {
                    'count': 0,
                    'values': [],
                    'category': self._find_unit_category(unit_key)
                }
            unit_counts[unit_key]['count'] += 1
            unit_counts[unit_key]['values'].append(unit.value)
        
        # Calculate statistics
        for unit_key, stats in unit_counts.items():
            values = stats['values']
            stats.update({
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'std': self._calculate_std(values)
            })
        
        return {
            'total_units_found': len(all_units),
            'unique_units': len(unit_counts),
            'unit_statistics': unit_counts,
            'recommendations': self._generate_unit_recommendations(unit_counts)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _generate_unit_recommendations(self, unit_stats: Dict[str, Any]) -> List[str]:
        """Generate unit usage recommendations"""
        recommendations = []
        
        # Check unit consistency
        categories = {}
        for unit, stats in unit_stats.items():
            category = stats['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(unit)
        
        for category, units in categories.items():
            if len(units) > 1:
                recommendations.append(
                    f"Found multiple units in {category} category: {', '.join(units)}, recommend unifying units"
                )
        
        # Check outliers
        for unit, stats in unit_stats.items():
            if stats['count'] > 10:  # 只有足够多的数据才检查异常值
                values = stats['values']
                mean = stats['mean']
                std = stats['std']
                
                outliers = [v for v in values if abs(v - mean) > 3 * std]
                if outliers:
                    recommendations.append(
                        f"Found {len(outliers)} outliers in unit {unit}, recommend checking data quality"
                    )
        
        return recommendations
    
    def suggest_unit_conversion(self, data: List[str]) -> Dict[str, Any]:
        """Suggest unit conversion"""
        logger.info("Analyzing unit conversion suggestions")
        
        unit_analysis = self.analyze_data_units(data)
        suggestions = []
        
        for unit, stats in unit_analysis['unit_statistics'].items():
            category = stats['category']
            
            if category in self.conversion_factors:
                target_units = list(self.conversion_factors[category].keys())
                if unit not in target_units:
                    # Recommend conversion to standard unit
                    standard_unit = target_units[0]
                    suggestions.append({
                        'from_unit': unit,
                        'to_unit': standard_unit,
                        'category': category,
                        'reason': f'Recommend converting to standard unit {standard_unit}'
                    })
        
        return {
            'conversion_suggestions': suggestions,
            'unit_analysis': unit_analysis
        }



