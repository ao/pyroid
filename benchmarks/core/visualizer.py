"""
Visualization utilities for the Pyroid benchmark suite.

This module provides utilities for generating visualizations from benchmark results.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

from .benchmark import Benchmark


class BenchmarkDashboard:
    """Class for generating a dashboard from benchmark results."""
    
    def __init__(self, benchmarks: List[Benchmark], output_dir: str = "benchmarks/dashboard"):
        """Initialize a benchmark dashboard.
        
        Args:
            benchmarks: The benchmarks to include in the dashboard.
            output_dir: The directory to write dashboard files to.
        """
        self.benchmarks = benchmarks
        self.output_dir = output_dir
        self.data = self._prepare_data()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        
        # Check for required dependencies
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly is not installed. Some visualizations will not be generated.")
            print("To install Plotly: pip install plotly")
        
        if not JINJA2_AVAILABLE:
            print("Warning: Jinja2 is not installed. HTML dashboard will not be generated.")
            print("To install Jinja2: pip install jinja2")
        
        if not WEASYPRINT_AVAILABLE:
            print("Warning: WeasyPrint is not installed. PDF dashboard will not be generated.")
            print("To install WeasyPrint: pip install weasyprint")
    
    def _prepare_data(self) -> pd.DataFrame:
        """Convert benchmark results to pandas DataFrame for easier visualization.
        
        Returns:
            A pandas DataFrame containing the benchmark results.
        """
        data = []
        
        for benchmark in self.benchmarks:
            comparisons = benchmark.compare_results()
            if not comparisons:
                continue
                
            for comp in comparisons:
                row = {
                    "benchmark_name": benchmark.name,
                    "benchmark_description": benchmark.description,
                    "implementation": comp["implementation"],
                    "duration_ms": comp["duration_ms"] if not comp["timed_out"] else None,
                    "timed_out": comp["timed_out"],
                    "timeout_seconds": comp["timeout_seconds"],
                    "display_time": comp["display_time"],
                    "speedup": comp["speedup"]
                }
                data.append(row)
                
        return pd.DataFrame(data)
    
    def generate_dashboard(self) -> None:
        """Generate the complete dashboard."""
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly is required for dashboard generation.")
            return
            
        # Generate all visualization components
        self._generate_performance_metrics()
        self._generate_comparison_charts()
        self._generate_scaling_charts()
        self._generate_real_world_scenarios()
        
        # Combine into HTML dashboard
        if JINJA2_AVAILABLE:
            self._generate_html_dashboard()
        
        # Generate PDF version
        if JINJA2_AVAILABLE and WEASYPRINT_AVAILABLE:
            self._generate_pdf_dashboard()
        
        print(f"Dashboard generated at {self.output_dir}/dashboard.html")
        if WEASYPRINT_AVAILABLE:
            print(f"PDF dashboard generated at {self.output_dir}/dashboard.pdf")
    
    def _generate_performance_metrics(self) -> None:
        """Generate key performance metrics panel."""
        if not PLOTLY_AVAILABLE:
            return
            
        # Calculate summary statistics
        pyroid_data = self.data[self.data["implementation"] == "pyroid"]
        python_data = self.data[self.data["implementation"] == "Python"]
        
        if pyroid_data.empty or python_data.empty:
            print("Warning: Not enough data to generate performance metrics.")
            return
        
        # Join the dataframes to calculate speedup
        merged_data = pd.merge(
            pyroid_data[["benchmark_name", "duration_ms"]],
            python_data[["benchmark_name", "duration_ms", "timed_out"]],
            on="benchmark_name",
            suffixes=("_pyroid", "_python")
        )
        
        # Calculate speedups where Python didn't time out
        valid_speedups = merged_data[~merged_data["timed_out_python"]]["duration_ms_python"] / merged_data[~merged_data["timed_out_python"]]["duration_ms_pyroid"]
        
        # Calculate metrics
        metrics = {
            "avg_speedup": valid_speedups.mean() if len(valid_speedups) > 0 else "N/A",
            "max_speedup": valid_speedups.max() if len(valid_speedups) > 0 else "N/A",
            "min_speedup": valid_speedups.min() if len(valid_speedups) > 0 else "N/A",
            "python_timeouts": python_data["timed_out"].sum(),
            "total_benchmarks": len(self.benchmarks)
        }
        
        # Create a figure for the metrics
        fig = go.Figure()
        
        # Add a table with the metrics
        fig.add_trace(go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color='#0066cc',
                align='center',
                font=dict(color='white', size=16)
            ),
            cells=dict(
                values=[
                    ["Average Speedup", "Maximum Speedup", "Minimum Speedup", "Python Timeouts", "Total Benchmarks"],
                    [
                        f"{metrics['avg_speedup']:.2f}x" if isinstance(metrics['avg_speedup'], (int, float)) else metrics['avg_speedup'],
                        f"{metrics['max_speedup']:.2f}x" if isinstance(metrics['max_speedup'], (int, float)) else metrics['max_speedup'],
                        f"{metrics['min_speedup']:.2f}x" if isinstance(metrics['min_speedup'], (int, float)) else metrics['min_speedup'],
                        f"{metrics['python_timeouts']} / {metrics['total_benchmarks']}",
                        metrics['total_benchmarks']
                    ]
                ],
                fill_color='#f9f9f9',
                align='center',
                font=dict(size=14)
            )
        ))
        
        fig.update_layout(
            title="Pyroid Performance Metrics",
            title_font=dict(size=24),
            width=800,
            height=400
        )
        
        # Save the figure
        fig.write_image(f"{self.output_dir}/images/performance_metrics.png", scale=2)
        fig.write_html(f"{self.output_dir}/images/performance_metrics.html")
    
    def _generate_comparison_charts(self) -> None:
        """Generate comparison visualizations."""
        if not PLOTLY_AVAILABLE:
            return
            
        # Filter for benchmarks where we have both Python and pyroid results
        benchmark_names = self.data[self.data["implementation"] == "pyroid"]["benchmark_name"].unique()
        
        # Create a DataFrame for the comparison chart
        comparison_data = []
        
        for benchmark_name in benchmark_names:
            pyroid_result = self.data[(self.data["benchmark_name"] == benchmark_name) &
                                     (self.data["implementation"] == "pyroid")]
            python_result = self.data[(self.data["benchmark_name"] == benchmark_name) &
                                     (self.data["implementation"] == "Python")]
            numpy_result = self.data[(self.data["benchmark_name"] == benchmark_name) &
                                    (self.data["implementation"] == "NumPy")]
            
            if not pyroid_result.empty and not python_result.empty:
                pyroid_time = pyroid_result.iloc[0]["duration_ms"]
                python_time = python_result.iloc[0]["duration_ms"] if not python_result.iloc[0]["timed_out"] else None
                numpy_time = numpy_result.iloc[0]["duration_ms"] if not numpy_result.empty and not numpy_result.iloc[0]["timed_out"] else None
                
                row = {
                    "benchmark_name": benchmark_name,
                    "pyroid_time": pyroid_time,
                    "python_time": python_time,
                    "numpy_time": numpy_time,
                    "python_timed_out": python_result.iloc[0]["timed_out"],
                    "numpy_timed_out": not numpy_result.empty and numpy_result.iloc[0]["timed_out"]
                }
                
                if python_time is not None:
                    row["speedup_vs_python"] = python_time / pyroid_time
                else:
                    row["speedup_vs_python"] = None
                    
                if numpy_time is not None:
                    row["speedup_vs_numpy"] = numpy_time / pyroid_time
                else:
                    row["speedup_vs_numpy"] = None
                    
                comparison_data.append(row)
        
        if not comparison_data:
            print("Warning: Not enough data to generate comparison charts.")
            return
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create speedup bar chart
        non_timeout_df = comparison_df[~comparison_df["python_timed_out"]]
        if not non_timeout_df.empty:
            fig = px.bar(
                non_timeout_df,
                x="benchmark_name",
                y="speedup_vs_python",
                title="Pyroid Speedup vs Python",
                labels={"benchmark_name": "Benchmark", "speedup_vs_python": "Speedup Factor (x)"},
                color_discrete_sequence=["#0066cc"]
            )
            
            # Add text labels on top of bars
            fig.update_traces(
                texttemplate='%{y:.1f}x',
                textposition='outside'
            )
            
            # Improve layout
            fig.update_layout(
                title_font=dict(size=24),
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickangle=-45,
                width=1000,
                height=600,
                margin=dict(t=100, b=100)
            )
            
            # Save the figure
            fig.write_image(f"{self.output_dir}/images/speedup_comparison.png", scale=2)
            fig.write_html(f"{self.output_dir}/images/speedup_comparison.html")
        
        # Create implementation comparison chart (log scale)
        valid_comparisons = comparison_df[~comparison_df["python_timed_out"]].copy()
        
        if not valid_comparisons.empty:
            # Melt the DataFrame for easier plotting
            melted_df = pd.melt(
                valid_comparisons,
                id_vars=["benchmark_name"],
                value_vars=["python_time", "numpy_time", "pyroid_time"],
                var_name="implementation",
                value_name="duration_ms"
            )
            
            # Clean up implementation names
            melted_df["implementation"] = melted_df["implementation"].str.replace("_time", "")
            
            # Remove NaN values (e.g., when NumPy wasn't tested)
            melted_df = melted_df.dropna(subset=["duration_ms"])
            
            # Create the bar chart
            fig = px.bar(
                melted_df,
                x="benchmark_name",
                y="duration_ms",
                color="implementation",
                title="Performance Comparison (Log Scale)",
                labels={"benchmark_name": "Benchmark", "duration_ms": "Duration (ms)", "implementation": "Implementation"},
                color_discrete_map={"python": "#1f77b4", "numpy": "#ff7f0e", "pyroid": "#2ca02c"},
                log_y=True,
                barmode="group"
            )
            
            # Add text labels on top of bars
            fig.update_traces(
                texttemplate='%{y:.1f}',
                textposition='outside'
            )
            
            # Improve layout
            fig.update_layout(
                title_font=dict(size=24),
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickangle=-45,
                width=1000,
                height=600,
                margin=dict(t=100, b=100)
            )
            
            # Save the figure
            fig.write_image(f"{self.output_dir}/images/implementation_comparison_log.png", scale=2)
            fig.write_html(f"{self.output_dir}/images/implementation_comparison_log.html")
            
            # Create a linear scale version too
            fig.update_layout(
                yaxis_type="linear",
                title="Performance Comparison (Linear Scale)"
            )
            
            fig.write_image(f"{self.output_dir}/images/implementation_comparison_linear.png", scale=2)
            fig.write_html(f"{self.output_dir}/images/implementation_comparison_linear.html")
    
    def _generate_scaling_charts(self) -> None:
        """Generate scaling visualizations."""
        if not PLOTLY_AVAILABLE:
            return
            
        # Find benchmarks that test the same operation with different sizes
        # For example, "Sum 1,000 numbers", "Sum 10,000 numbers", etc.
        
        # Extract benchmark names and try to parse the operation and size
        benchmark_info = []
        for benchmark_name in self.data["benchmark_name"].unique():
            # Try to match patterns like "Sum 1,000 numbers" or "Regex replace 1,000,000 chars"
            match = re.match(r"([A-Za-z\s]+)([0-9,]+)([A-Za-z\s]+)", benchmark_name)
            if match:
                operation = match.group(1).strip()
                size_str = match.group(2).replace(",", "")
                unit = match.group(3).strip()
                
                try:
                    size = int(size_str)
                    benchmark_info.append({
                        "benchmark_name": benchmark_name,
                        "operation": operation,
                        "size": size,
                        "unit": unit
                    })
                except ValueError:
                    continue
        
        if not benchmark_info:
            print("Warning: No scaling benchmarks found.")
            return
            
        # Convert to DataFrame
        benchmark_info_df = pd.DataFrame(benchmark_info)
        
        # Group by operation
        operations = benchmark_info_df["operation"].unique()
        
        for operation in operations:
            # Get benchmarks for this operation
            operation_benchmarks = benchmark_info_df[benchmark_info_df["operation"] == operation]
            
            if len(operation_benchmarks) < 2:
                # Need at least 2 sizes to show scaling
                continue
                
            # Get the unit from the first benchmark
            unit = operation_benchmarks.iloc[0]["unit"]
            
            # Create a DataFrame for the scaling chart
            scaling_data = []
            
            for _, benchmark_row in operation_benchmarks.iterrows():
                benchmark_name = benchmark_row["benchmark_name"]
                size = benchmark_row["size"]
                
                pyroid_result = self.data[(self.data["benchmark_name"] == benchmark_name) &
                                         (self.data["implementation"] == "pyroid")]
                python_result = self.data[(self.data["benchmark_name"] == benchmark_name) &
                                         (self.data["implementation"] == "Python")]
                numpy_result = self.data[(self.data["benchmark_name"] == benchmark_name) &
                                        (self.data["implementation"] == "NumPy")]
                
                if not pyroid_result.empty:
                    pyroid_time = pyroid_result.iloc[0]["duration_ms"]
                    python_time = python_result.iloc[0]["duration_ms"] if not python_result.empty and not python_result.iloc[0]["timed_out"] else None
                    numpy_time = numpy_result.iloc[0]["duration_ms"] if not numpy_result.empty and not numpy_result.iloc[0]["timed_out"] else None
                    
                    row = {
                        "size": size,
                        "pyroid_time": pyroid_time,
                        "python_time": python_time,
                        "numpy_time": numpy_time
                    }
                    
                    if python_time is not None and pyroid_time is not None:
                        row["speedup_vs_python"] = python_time / pyroid_time
                    else:
                        row["speedup_vs_python"] = None
                        
                    if numpy_time is not None and pyroid_time is not None:
                        row["speedup_vs_numpy"] = numpy_time / pyroid_time
                    else:
                        row["speedup_vs_numpy"] = None
                        
                    scaling_data.append(row)
            
            if not scaling_data:
                continue
                
            scaling_df = pd.DataFrame(scaling_data)
            scaling_df = scaling_df.sort_values("size")
            
            # Create scaling chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f"Performance Scaling for {operation}{unit}",
                    f"Speedup Scaling for {operation}{unit}"
                ),
                vertical_spacing=0.15
            )
            
            # Add performance lines
            if not scaling_df["pyroid_time"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=scaling_df["size"],
                        y=scaling_df["pyroid_time"],
                        mode="lines+markers",
                        name="pyroid",
                        line=dict(color="#2ca02c", width=3)
                    ),
                    row=1, col=1
                )
            
            if not scaling_df["numpy_time"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=scaling_df["size"],
                        y=scaling_df["numpy_time"],
                        mode="lines+markers",
                        name="NumPy",
                        line=dict(color="#ff7f0e", width=3)
                    ),
                    row=1, col=1
                )
            
            if not scaling_df["python_time"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=scaling_df["size"],
                        y=scaling_df["python_time"],
                        mode="lines+markers",
                        name="Python",
                        line=dict(color="#1f77b4", width=3)
                    ),
                    row=1, col=1
                )
            
            # Add speedup lines
            if not scaling_df["speedup_vs_python"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=scaling_df["size"],
                        y=scaling_df["speedup_vs_python"],
                        mode="lines+markers",
                        name="vs Python",
                        line=dict(color="#d62728", width=3)
                    ),
                    row=2, col=1
                )
            
            if not scaling_df["speedup_vs_numpy"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=scaling_df["size"],
                        y=scaling_df["speedup_vs_numpy"],
                        mode="lines+markers",
                        name="vs NumPy",
                        line=dict(color="#9467bd", width=3)
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"Scaling Analysis: {operation}{unit}",
                title_font=dict(size=24),
                width=1000,
                height=800,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update x and y axes
            fig.update_xaxes(
                title_text=f"Size ({unit})",
                type="log",
                row=1, col=1
            )
            
            fig.update_yaxes(
                title_text="Duration (ms)",
                type="log",
                row=1, col=1
            )
            
            fig.update_xaxes(
                title_text=f"Size ({unit})",
                type="log",
                row=2, col=1
            )
            
            fig.update_yaxes(
                title_text="Speedup Factor (x)",
                row=2, col=1
            )
            
            # Save the figure
            safe_operation = operation.replace(" ", "_").lower()
            fig.write_image(f"{self.output_dir}/images/scaling_{safe_operation}.png", scale=2)
            fig.write_html(f"{self.output_dir}/images/scaling_{safe_operation}.html")
    
    def _generate_real_world_scenarios(self) -> None:
        """Generate real-world scenario visualizations."""
        if not PLOTLY_AVAILABLE:
            return
            
        # Filter for real-world scenario benchmarks
        scenarios = [
            "Data Processing Pipeline",
            "Web Scraping",
            "Text Processing Pipeline",
            "Scientific Computing"
        ]
        
        scenario_data = []
        
        for scenario in scenarios:
            # Find benchmarks that match or contain the scenario name
            matching_benchmarks = [b for b in self.benchmarks
                                  if scenario.lower() in b.name.lower()]
            
            for benchmark in matching_benchmarks:
                comparisons = benchmark.compare_results()
                if not comparisons:
                    continue
                    
                pyroid_result = next((c for c in comparisons if c["implementation"] == "pyroid"), None)
                python_result = next((c for c in comparisons if c["implementation"] == "Python"), None)
                
                if pyroid_result and python_result:
                    row = {
                        "scenario": benchmark.name,
                        "pyroid_time": pyroid_result["duration_ms"] if not pyroid_result["timed_out"] else None,
                        "python_time": python_result["duration_ms"] if not python_result["timed_out"] else None,
                        "python_timed_out": python_result["timed_out"],
                        "pyroid_timed_out": pyroid_result["timed_out"]
                    }
                    
                    if not python_result["timed_out"] and not pyroid_result["timed_out"]:
                        row["speedup"] = python_result["duration_ms"] / pyroid_result["duration_ms"]
                    else:
                        row["speedup"] = None
                        
                    scenario_data.append(row)
        
        if not scenario_data:
            print("Warning: No real-world scenario benchmarks found.")
            return
            
        scenario_df = pd.DataFrame(scenario_data)
        
        # Create bar chart for real-world scenarios
        fig = go.Figure()
        
        # Add pyroid bars
        fig.add_trace(go.Bar(
            x=scenario_df["scenario"],
            y=scenario_df["pyroid_time"],
            name="pyroid",
            marker_color="#2ca02c"
        ))
        
        # Add Python bars for non-timed-out scenarios
        non_timeout_df = scenario_df[~scenario_df["python_timed_out"]]
        if not non_timeout_df.empty:
            fig.add_trace(go.Bar(
                x=non_timeout_df["scenario"],
                y=non_timeout_df["python_time"],
                name="Python",
                marker_color="#1f77b4"
            ))
        
        # Add annotations for speedups
        for i, row in scenario_df.iterrows():
            if row["speedup"] is not None:
                fig.add_annotation(
                    x=row["scenario"],
                    y=max(row["pyroid_time"], row["python_time"]) * 1.1,
                    text=f"{row['speedup']:.1f}x faster",
                    showarrow=False,
                    font=dict(size=14, color="#d62728")
                )
            elif row["python_timed_out"]:
                fig.add_annotation(
                    x=row["scenario"],
                    y=row["pyroid_time"] * 1.1,
                    text="Python timed out",
                    showarrow=False,
                    font=dict(size=14, color="#d62728")
                )
        
        # Update layout
        fig.update_layout(
            title="Real-world Scenario Performance",
            title_font=dict(size=24),
            xaxis_title="Scenario",
            yaxis_title="Duration (ms)",
            xaxis_tickangle=-45,
            width=1000,
            height=600,
            barmode="group"
        )
        
        # Save the figure
        fig.write_image(f"{self.output_dir}/images/real_world_scenarios.png", scale=2)
        fig.write_html(f"{self.output_dir}/images/real_world_scenarios.html")
    
    def _generate_html_dashboard(self) -> None:
        """Generate HTML dashboard combining all visualizations."""
        if not JINJA2_AVAILABLE:
            return
            
        # HTML template for the dashboard
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pyroid Performance Dashboard</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .header {
                    background-color: #0066cc;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .section {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 20px;
                }
                .section-title {
                    color: #0066cc;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                    margin-top: 0;
                }
                .chart-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .chart {
                    max-width: 100%;
                    height: auto;
                }
                .footer {
                    text-align: center;
                    padding: 20px;
                    color: #666;
                    font-size: 14px;
                }
                .two-column {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }
                .column {
                    flex: 1;
                    min-width: 300px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Pyroid Performance Dashboard</h1>
                <p>Benchmark results demonstrating Pyroid's performance advantages</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2 class="section-title">Performance Summary</h2>
                    <div class="chart-container">
                        <img class="chart" src="images/performance_metrics.png" alt="Performance Metrics">
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Speedup Comparison</h2>
                    <div class="chart-container">
                        <img class="chart" src="images/speedup_comparison.png" alt="Speedup Comparison">
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Implementation Comparison</h2>
                    <div class="two-column">
                        <div class="column">
                            <div class="chart-container">
                                <h3>Log Scale</h3>
                                <img class="chart" src="images/implementation_comparison_log.png" alt="Implementation Comparison (Log Scale)">
                            </div>
                        </div>
                        <div class="column">
                            <div class="chart-container">
                                <h3>Linear Scale</h3>
                                <img class="chart" src="images/implementation_comparison_linear.png" alt="Implementation Comparison (Linear Scale)">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Scaling Analysis</h2>
                    {% for chart in scaling_charts %}
                    <div class="chart-container">
                        <img class="chart" src="{{ chart }}" alt="Scaling Analysis">
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2 class="section-title">Real-world Scenarios</h2>
                    <div class="chart-container">
                        <img class="chart" src="images/real_world_scenarios.png" alt="Real-world Scenarios">
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by Pyroid Benchmark Suite on {{ date }}</p>
            </div>
        </body>
        </html>
        """
        
        # Get scaling chart filenames
        scaling_charts = []
        if os.path.exists(f"{self.output_dir}/images"):
            for filename in os.listdir(f"{self.output_dir}/images"):
                if filename.startswith("scaling_") and filename.endswith(".png"):
                    scaling_charts.append(f"images/{filename}")
        
        # Render the template
        template = Template(html_template)
        html = template.render(
            scaling_charts=scaling_charts,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Write the HTML file
        with open(f"{self.output_dir}/dashboard.html", "w") as f:
            f.write(html)
    
    def _generate_pdf_dashboard(self) -> None:
        """Generate PDF version of the dashboard."""
        if not WEASYPRINT_AVAILABLE or not JINJA2_AVAILABLE:
            return
            
        try:
            # Convert HTML to PDF
            html_path = f"{self.output_dir}/dashboard.html"
            pdf_path = f"{self.output_dir}/dashboard.pdf"
            
            # Check if HTML file exists
            if not os.path.exists(html_path):
                print(f"Warning: HTML dashboard not found at {html_path}")
                return
                
            # Generate PDF
            weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
        except Exception as e:
            print(f"Error generating PDF dashboard: {e}")
