"""
Visualization Engine - Generate various types of charts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
import io
import base64

class VisualizationEngine:
    """Visualization Engine"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualization engine"""
        self.style = style
        plt.style.use(style)
        
        # Set font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Color configuration
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
    
    def create_histogram(self, df: pd.DataFrame, column: str, 
                        bins: int = 30, title: str = None) -> Dict[str, Any]:
        """Create histogram"""
        logger.info(f"Creating histogram: {column}")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get column data and filter None/NaN
            col_data = df[column]
            if isinstance(col_data, list):
                # SimpleDataFrame columns are list
                values = [x for x in col_data if x is not None and str(x).lower() != 'nan']
            else:
                # pandas DataFrame
                values = col_data.dropna()
            
            if not values:
                return {"error": "No valid data in column"}
            
            # Create histogram
            ax.hist(values, bins=bins, alpha=0.7, 
                   color=self.colors['primary'], edgecolor='black')
            
            # Set title and labels
            title = title or f"{column} Distribution"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            
            # Add statistical info
            import numpy as np
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', 
                      label=f'Median: {median_val:.2f}')
            ax.legend()
            
            # Convert to base64
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "histogram",
                "title": title,
                "image": img_base64,
                "statistics": {
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to create histogram: {e}")
            return {"error": str(e)}
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                           color_col: str = None, title: str = None) -> Dict[str, Any]:
        """Create scatter plot"""
        logger.info(f"Creating scatter plot: {x_col} vs {y_col}")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get and filter data
            x_data = df[x_col] if not isinstance(df[x_col], list) else df[x_col]
            y_data = df[y_col] if not isinstance(df[y_col], list) else df[y_col]
            
            # Filter None and NaN
            valid_indices = []
            x_values = []
            y_values = []
            for i in range(len(x_data)):
                x_val = x_data[i]
                y_val = y_data[i]
                if x_val is not None and y_val is not None:
                    if str(x_val).lower() != 'nan' and str(y_val).lower() != 'nan':
                        try:
                            x_values.append(float(x_val))
                            y_values.append(float(y_val))
                            valid_indices.append(i)
                        except:
                            pass
            
            if not x_values or not y_values:
                return {"error": "No valid numeric data"}
            
            if color_col and color_col in df.columns:
                # Group by color
                color_data = df[color_col]
                unique_values = list(set(color_data[i] for i in valid_indices))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
                
                for idx, value in enumerate(unique_values):
                    x_subset = [x_values[i] for i, vi in enumerate(valid_indices) if color_data[vi] == value]
                    y_subset = [y_values[i] for i, vi in enumerate(valid_indices) if color_data[vi] == value]
                    ax.scatter(x_subset, y_subset, 
                             c=[colors[idx]], label=str(value), alpha=0.7)
                ax.legend()
            else:
                ax.scatter(x_values, y_values, alpha=0.7, 
                          color=self.colors['primary'])
            
            # Set title and labels
            title = title or f"{x_col} vs {y_col} Scatter Plot"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            
            # Add trend line
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            ax.plot(x_values, p(x_values), "r--", alpha=0.8)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "scatter",
                "title": title,
                "image": img_base64,
                "correlation": float(correlation)
            }
            
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {e}")
            return {"error": str(e)}
    
    def create_line_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                        title: str = None) -> Dict[str, Any]:
        """Create line plot"""
        logger.info(f"Creating line plot: {x_col} vs {y_col}")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sort data
            sorted_df = df.sort_values(x_col)
            
            ax.plot(sorted_df[x_col], sorted_df[y_col], 
                   color=self.colors['primary'], linewidth=2, marker='o', markersize=4)
            
            title = title or f"{y_col} Trend Over Time"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "line",
                "title": title,
                "image": img_base64
            }
            
        except Exception as e:
            logger.error(f"Failed to create line plot: {e}")
            return {"error": str(e)}
    
    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                          title: str = None) -> Dict[str, Any]:
        """Create bar chart"""
        logger.info(f"Creating bar chart: {x_col} vs {y_col}")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group aggregation
            grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
            
            bars = ax.bar(range(len(grouped)), grouped.values, 
                         color=self.colors['primary'], alpha=0.8)
            
            # Set labels
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            
            title = title or f"{y_col} Comparison by {x_col}"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(y_col, fontsize=12)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "bar",
                "title": title,
                "image": img_base64,
                "data": grouped
            }
            
        except Exception as e:
            logger.error(f"Failed to create bar chart: {e}")
            return {"error": str(e)}
    
    def create_pie_chart(self, df: pd.DataFrame, column: str, 
                        title: str = None) -> Dict[str, Any]:
        """Create pie chart"""
        logger.info(f"Creating pie chart: {column}")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Get column data
            col_data = df[column]
            
            # Calculate proportions
            from collections import Counter
            value_counts = Counter(col_data)
            
            if not value_counts:
                return {"error": "No data in column"}
            
            # Create pie chart
            labels = list(value_counts.keys())
            values = list(value_counts.values())
            
            wedges, texts, autotexts = ax.pie(values, 
                                            labels=labels,
                                            autopct='%1.1f%%',
                                            startangle=90)
            
            title = title or f"{column} Distribution"
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "pie",
                "title": title,
                "image": img_base64,
                "data": dict(value_counts)
            }
            
        except Exception as e:
            logger.error(f"Failed to create pie chart: {e}")
            return {"error": str(e)}
    
    def create_heatmap(self, df: pd.DataFrame, columns: List[str] = None,
                      title: str = None) -> Dict[str, Any]:
        """Create heatmap"""
        logger.info("Creating heatmap")
        
        try:
            # Select numeric columns
            if columns:
                data = df[columns]
            else:
                data = df.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add value labels
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation', rotation=270, labelpad=20)
            
            title = title or "Variable Correlation Heatmap"
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "heatmap",
                "title": title,
                "image": img_base64,
                "correlation_matrix": corr_matrix
            }
            
        except Exception as e:
            logger.error(f"Failed to create heatmap: {e}")
            return {"error": str(e)}
    
    def create_box_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                       title: str = None) -> Dict[str, Any]:
        """Create box plot"""
        logger.info(f"Creating box plot: {x_col} vs {y_col}")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create box plot
            df.boxplot(column=y_col, by=x_col, ax=ax)
            
            title = title or f"{y_col} Distribution by {x_col}"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            
            plt.tight_layout()
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "box",
                "title": title,
                "image": img_base64
            }
            
        except Exception as e:
            logger.error(f"Failed to create box plot: {e}")
            return {"error": str(e)}
    
    def create_interactive_plot(self, df: pd.DataFrame, plot_type: str,
                              x_col: str, y_col: str, **kwargs) -> Dict[str, Any]:
        """Create interactive plot"""
        logger.info(f"Creating interactive plot: {plot_type}")
        
        try:
            if plot_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, **kwargs)
            elif plot_type == "line":
                fig = px.line(df, x=x_col, y=y_col, **kwargs)
            elif plot_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, **kwargs)
            elif plot_type == "histogram":
                fig = px.histogram(df, x=x_col, **kwargs)
            elif plot_type == "box":
                fig = px.box(df, x=x_col, y=y_col, **kwargs)
            else:
                raise ValueError(f"Unsupported chart type: {plot_type}")
            
            # Convert to HTML
            html = fig.to_html(include_plotlyjs='cdn')
            
            return {
                "type": f"interactive_{plot_type}",
                "html": html,
                "plotly_json": fig.to_json()
            }
            
        except Exception as e:
            logger.error(f"Failed to create interactive plot: {e}")
            return {"error": str(e)}
    
    def auto_generate_visualizations(self, df: pd.DataFrame, 
                                   query: str = None) -> List[Dict[str, Any]]:
        """Auto generate visualizations"""
        logger.info("Auto generating visualizations")
        
        visualizations = []
        
        # Get numeric and categorical columns
        # SimpleDataFrame columns is already list, no need for tolist()
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        categorical_df = df.select_dtypes(include=['object'])
        numeric_cols = numeric_df.columns if hasattr(numeric_df, 'columns') else []
        categorical_cols = categorical_df.columns if hasattr(categorical_df, 'columns') else []
        
        # 1. Numeric column distribution
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            try:
                hist_result = self.create_histogram(df, col)
                if "error" not in hist_result:
                    visualizations.append(hist_result)
            except:
                pass
        
        # 2. Correlation heatmap
        if len(numeric_cols) > 1:
            try:
                heatmap_result = self.create_heatmap(df, numeric_cols[:5])
                if "error" not in heatmap_result:
                    visualizations.append(heatmap_result)
            except:
                pass
        
        # 3. Categorical variable distribution
        for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            try:
                pie_result = self.create_pie_chart(df, col)
                if "error" not in pie_result:
                    visualizations.append(pie_result)
            except:
                pass
        
        # 4. Numeric variable relationship
        if len(numeric_cols) >= 2:
            try:
                scatter_result = self.create_scatter_plot(df, numeric_cols[0], numeric_cols[1])
                if "error" not in scatter_result:
                    visualizations.append(scatter_result)
            except:
                pass
        
        return visualizations
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return img_base64
    
    def get_visualization_suggestions(self, df: pd.DataFrame, 
                                    query: str = None) -> List[Dict[str, Any]]:
        """Get visualization suggestions"""
        suggestions = []
        
        # SimpleDataFrame columns is already list, no need for tolist()
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        categorical_df = df.select_dtypes(include=['object'])
        numeric_cols = numeric_df.columns if hasattr(numeric_df, 'columns') else []
        categorical_cols = categorical_df.columns if hasattr(categorical_df, 'columns') else []
        
        # Data type based suggestions
        if len(numeric_cols) > 0:
            suggestions.append({
                "type": "histogram",
                "description": "Numeric variable distribution analysis",
                "columns": numeric_cols[:1],
                "reason": "Understand the distribution characteristics of numeric variables"
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "description": "Variable relationship analysis",
                "columns": numeric_cols[:2],
                "reason": "Explore correlations between numeric variables"
            })
        
        if len(categorical_cols) > 0:
            suggestions.append({
                "type": "pie",
                "description": "Categorical variable distribution",
                "columns": categorical_cols[:1],
                "reason": "Understand the distribution of categorical variables"
            })
        
        if len(numeric_cols) > 1:
            suggestions.append({
                "type": "heatmap",
                "description": "Correlation analysis",
                "columns": numeric_cols,
                "reason": "Comprehensive understanding of relationships between variables"
            })
        
        return suggestions


