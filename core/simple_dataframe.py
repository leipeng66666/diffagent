#!/usr/bin/env python3
"""
Simple DataFrame alternative implementation to avoid pandas compatibility issues
"""

from typing import List, Dict, Any, Union
import csv
import json

# Unicode infinity symbol '∞' (U+221E) is not a valid Python float literal.
# Normalize it to 'inf' so float() can parse it.
def _safe_float(s: str) -> float:
    """Convert string to float, normalizing Unicode infinity → 'inf'."""
    return float(str(s).replace('∞', 'inf'))


class SimpleDataFrame:
    """Simple DataFrame alternative implementation"""
    
    def __init__(self, data: Union[Dict[str, List], List[Dict]] = None):
        """Initialize DataFrame"""
        if data is None:
            self.data = {}
            self.columns = []
        elif isinstance(data, dict):
            self.data = data
            self.columns = list(data.keys())
        elif isinstance(data, list) and len(data) > 0:
            # Create from list of dictionaries
            self.columns = list(data[0].keys())
            self.data = {col: [row[col] for row in data] for col in self.columns}
        else:
            self.data = {}
            self.columns = []
    
    @property
    def shape(self) -> tuple:
        """Return data shape"""
        if not self.columns:
            return (0, 0)
        return (len(self.data[self.columns[0]]), len(self.columns))
    
    def head(self, n: int = 5) -> 'SimpleDataFrame':
        """Return first n rows"""
        if not self.columns:
            return SimpleDataFrame()
        
        new_data = {}
        for col in self.columns:
            new_data[col] = self.data[col][:n]
        
        return SimpleDataFrame(new_data)
    
    def fillna(self, value: str = '') -> 'SimpleDataFrame':
        """Fill missing values"""
        new_data = {}
        for col in self.columns:
            new_data[col] = [str(val) if val is not None and val != '' else value for val in self.data[col]]
        
        return SimpleDataFrame(new_data)
    
    def astype(self, dtype: str) -> 'SimpleDataFrame':
        """Convert data type"""
        if dtype == 'str':
            new_data = {}
            for col in self.columns:
                new_data[col] = [str(val) for val in self.data[col]]
            return SimpleDataFrame(new_data)
        return self
    
    def iterrows(self):
        """Iterate rows"""
        if not self.columns:
            return
        
        for i in range(len(self.data[self.columns[0]])):
            row_data = {col: self.data[col][i] for col in self.columns}
            yield i, SimpleRow(row_data)
    
    def __getitem__(self, key):
        """Get column data"""
        if isinstance(key, str):
            return self.data[key]
        return self
    
    def __setitem__(self, key, value):
        """Set column data"""
        if isinstance(key, str):
            self.data[key] = value
            if key not in self.columns:
                self.columns.append(key)
    
    def __str__(self):
        """String representation"""
        if not self.columns:
            return "Empty DataFrame"
        
        # Create table string
        lines = []
        
        # Header
        header = " | ".join(f"{col:>10}" for col in self.columns)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data rows
        for i in range(min(5, len(self.data[self.columns[0]]))):
            row = " | ".join(f"{str(self.data[col][i]):>10}" for col in self.columns)
            lines.append(row)
        
        if len(self.data[self.columns[0]]) > 5:
            lines.append("...")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        """Return row count"""
        if not self.columns:
            return 0
        return len(self.data[self.columns[0]])
    
    def copy(self):
        """Return a copy of the DataFrame"""
        if not self.columns:
            return SimpleDataFrame()
        
        new_data = {}
        for col in self.columns:
            new_data[col] = self.data[col].copy()
        
        return SimpleDataFrame(new_data)
    
    @property
    def dtypes(self):
        """Return data type information"""
        if not self.columns:
            return {}
        
        # Simple data type detection
        dtypes = {}
        for col in self.columns:
            if not self.data[col]:
                dtypes[col] = 'object'
                continue
            
            # Check if it's a numeric type
            try:
                _safe_float(self.data[col][0])
                dtypes[col] = 'float64'
            except (ValueError, TypeError):
                dtypes[col] = 'object'
        
        return dtypes
    
    @property
    def index(self):
        """Return index"""
        if not self.columns:
            return []
        return list(range(len(self.data[self.columns[0]])))
    
    def isna(self):
        """Check for missing values"""
        if not self.columns:
            return SimpleDataFrame()
        
        result = {}
        for col in self.columns:
            result[col] = [val is None or val == '' or val == 'nan' for val in self.data[col]]
        
        return SimpleDataFrame(result)
    
    def isnull(self):
        """Check for null values (same as isna)"""
        return self.isna()
    
    def dropna(self):
        """Drop rows containing missing values"""
        if not self.columns:
            return SimpleDataFrame()
        
        # Find rows without missing values
        valid_rows = []
        for i in range(len(self.data[self.columns[0]])):
            has_null = False
            for col in self.columns:
                val = self.data[col][i]
                if val is None or val == '' or val == 'nan':
                    has_null = True
                    break
            if not has_null:
                valid_rows.append(i)
        
        # 创建新的DataFrame
        new_data = {}
        for col in self.columns:
            new_data[col] = [self.data[col][i] for i in valid_rows]
        
        return SimpleDataFrame(new_data)
    
    def to_dict(self, orient='records'):
        """Convert to dictionary"""
        if orient == 'records':
            result = []
            for i in range(len(self.data[self.columns[0]])):
                row = {col: self.data[col][i] for col in self.columns}
                result.append(row)
            return result
        else:
            return self.data
    
    def values(self):
        """Return numeric array"""
        if not self.columns:
            return []
        
        result = []
        for i in range(len(self.data[self.columns[0]])):
            row = [self.data[col][i] for col in self.columns]
            result.append(row)
        return result
    
    def value_counts(self):
        """Return value counts"""
        if not self.columns:
            return SimpleDataFrame()
        
        # Calculate value counts for each column
        counts = {}
        for col in self.columns:
            col_counts = {}
            for val in self.data[col]:
                val_str = str(val)
                col_counts[val_str] = col_counts.get(val_str, 0) + 1
            counts[col] = col_counts
        
        return SimpleDataFrame(counts)
    
    def value_counts_single(self, column):
        """Return value counts for a single column"""
        if column not in self.data:
            return SimpleDataFrame()
        
        col_counts = {}
        for val in self.data[column]:
            val_str = str(val)
            col_counts[val_str] = col_counts.get(val_str, 0) + 1
        
        # Convert to list format
        result = []
        for value, count in col_counts.items():
            result.append({'value': value, 'count': count})
        
        return result
    
    def _safe_numeric_values(self, col: str) -> list:
        """Safely extract numeric values from a column, skipping non-numeric strings."""
        values = []
        for val in self.data[col]:
            if val is None or val == '':
                continue
            try:
                values.append(_safe_float(val))
            except (ValueError, TypeError):
                continue
        return values

    def describe(self):
        """Return descriptive statistics"""
        if not self.columns:
            return {}

        # Only calculate statistics for numeric columns
        numeric_cols = []
        for col in self.columns:
            try:
                # Try to convert first value to numeric
                _safe_float(self.data[col][0])
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue

        if not numeric_cols:
            return {}

        # Calculate statistics
        stats = {}
        for col in numeric_cols:
            values = self._safe_numeric_values(col)
            if values:
                stats[col] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                    'min': min(values),
                    'max': max(values)
                }
        
        return stats
    
    def corr(self):
        """Return correlation coefficient matrix"""
        if not self.columns:
            return SimpleDataFrame()
        
        # Only calculate correlation for numeric columns
        numeric_cols = []
        for col in self.columns:
            try:
                _safe_float(self.data[col][0])
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        if len(numeric_cols) < 2:
            return SimpleDataFrame()
        
        # Simple correlation coefficient calculation
        corr_matrix = {}
        for col1 in numeric_cols:
            corr_matrix[col1] = {}
            for col2 in numeric_cols:
                if col1 == col2:
                    corr_matrix[col1][col2] = 1.0
                else:
                    # Simplified correlation calculation
                    values1 = self._safe_numeric_values(col1)
                    values2 = self._safe_numeric_values(col2)
                    
                    if len(values1) == len(values2) and len(values1) > 1:
                        mean1 = sum(values1) / len(values1)
                        mean2 = sum(values2) / len(values2)
                        
                        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
                        denominator = (sum((x - mean1)**2 for x in values1) * sum((y - mean2)**2 for y in values2))**0.5
                        
                        if denominator != 0:
                            corr_matrix[col1][col2] = numerator / denominator
                        else:
                            corr_matrix[col1][col2] = 0.0
                    else:
                        corr_matrix[col1][col2] = 0.0
        
        return SimpleDataFrame(corr_matrix)
    
    def sum(self):
        """Return sum of each column"""
        if not self.columns:
            return {}

        result = {}
        for col in self.columns:
            values = self._safe_numeric_values(col)
            result[col] = sum(values) if values else 0

        return result
    
    def mean(self):
        """Return mean of each column"""
        if not self.columns:
            return {}

        result = {}
        for col in self.columns:
            values = self._safe_numeric_values(col)
            result[col] = sum(values) / len(values) if values else 0

        return result
    
    def std(self):
        """Return standard deviation of each column"""
        if not self.columns:
            return {}

        result = {}
        for col in self.columns:
            values = self._safe_numeric_values(col)
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val)**2 for x in values) / len(values)
                result[col] = variance**0.5
            else:
                result[col] = 0

        return result
    
    def min(self):
        """Return minimum of each column"""
        if not self.columns:
            return {}

        result = {}
        for col in self.columns:
            values = self._safe_numeric_values(col)
            result[col] = min(values) if values else None

        return result

    def max(self):
        """Return maximum of each column"""
        if not self.columns:
            return {}

        result = {}
        for col in self.columns:
            values = self._safe_numeric_values(col)
            result[col] = max(values) if values else None

        return result
    
    def nunique(self):
        """Return count of unique values for each column"""
        if not self.columns:
            return {}
        
        result = {}
        for col in self.columns:
            unique_values = set(str(val) for val in self.data[col] if val != '' and val is not None)
            result[col] = len(unique_values)
        
        return result
    
    def select_dtypes(self, include=None, exclude=None):
        """Select columns of specific data types"""
        if not self.columns:
            return SimpleDataFrame()

        import numpy as np

        # Normalize type specs to strings — col_type from dtypes is 'float64'/'object',
        # but callers may pass numpy classes like np.number or np.float64.
        def _normalize(spec):
            """Convert numpy type classes to their string dtype names."""
            if spec is np.number:
                return {'float64', 'int64', 'float32', 'int32', 'float16', 'int16',
                        'int8', 'uint8', 'uint16', 'uint32', 'uint64', 'complex64', 'complex128'}
            if isinstance(spec, type):
                try:
                    return {np.dtype(spec).name}
                except TypeError:
                    pass
            if isinstance(spec, str):
                return {spec}
            return set()

        selected_cols = []
        for col in self.columns:
            col_type = self.dtypes.get(col, 'object')

            if include is not None:
                allowed = set()
                for inc in include:
                    allowed |= _normalize(inc)
                if col_type in allowed:
                    selected_cols.append(col)
            elif exclude is not None:
                forbidden = set()
                for exc in exclude:
                    forbidden |= _normalize(exc)
                if col_type not in forbidden:
                    selected_cols.append(col)
            else:
                selected_cols.append(col)

        # 创建新的DataFrame
        new_data = {col: self.data[col] for col in selected_cols}
        return SimpleDataFrame(new_data)

class SimpleRow:
    """Simple row object"""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

class SimpleDataLoader:
    """Simple data loader"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
    
    def load_table(self, file_path: str) -> SimpleDataFrame:
        """Load table data"""
        print(f"Loading table file: {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                return self._load_csv(file_path)
            elif file_path.endswith('.json'):
                return self._load_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            print(f"Failed to load table: {e}")
            raise
    
    def _load_csv(self, file_path: str) -> SimpleDataFrame:
        """Load CSV file"""
        # Try encodings in order: utf-8-sig (handles BOM), utf-8, latin-1, cp1252
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        last_error = None
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
        else:
            raise UnicodeDecodeError(f"Could not decode file with any of {encodings}: {last_error}")
        
        if not rows:
            raise ValueError("File is empty")
        
        headers = rows[0]
        data_rows = rows[1:]
        
        # Create data dictionary
        data = {}
        for i, header in enumerate(headers):
            col_data = []
            for row in data_rows:
                if i < len(row):
                    # Normalize Unicode infinity symbol '∞' → 'inf' so float() can parse it
                    val = str(row[i]).replace('∞', 'inf')
                    col_data.append(val)
                else:
                    col_data.append('')
            data[header] = col_data
        
        return SimpleDataFrame(data)
    
    def _load_json(self, file_path: str) -> SimpleDataFrame:
        """Load JSON file"""
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        last_error = None
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    data = json.load(f)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
        else:
            raise UnicodeDecodeError(f"Could not decode file with any of {encodings}: {last_error}")
        
        if isinstance(data, list) and len(data) > 0:
            return SimpleDataFrame(data)
        else:
            raise ValueError("JSON format not supported")

def test_simple_dataframe():
    """Test SimpleDataFrame"""
    print("Test SimpleDataFrame")
    print("=" * 50)
    
    # Test 1: Create DataFrame
    print("Test 1: Create DataFrame")
    data = {'Name': ['Zhang San', 'Li Si'], 'Age': [25, 30]}
    df = SimpleDataFrame(data)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("Data:")
    print(df)
    
    # Test 2: Load CSV
    print("\nTest 2: Load CSV")
    loader = SimpleDataLoader()
    df2 = loader.load_table('test_data.csv')
    print(f"Shape: {df2.shape}")
    print("Data:")
    print(df2)
    
    # Test 3: Iterate rows
    print("\nTest 3: Iterate rows")
    for i, row in df2.iterrows():
        print(f"Row {i}: {row.data}")
        if i >= 2:  # Only show first 3 rows
            break
    
    return True

if __name__ == "__main__":
    test_simple_dataframe()
