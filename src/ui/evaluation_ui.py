"""
Evaluation UI Handlers

Handles UI interactions for the evaluation tab including:
- Loading and displaying evaluation results
- Running quick evaluations
- Opening evaluation reports
"""

import os
import json
import logging
import threading
import subprocess
import sys
from typing import Optional, Dict, Any
import tkinter as tk

# Get the Python executable path (cross-platform)
PYTHON_EXECUTABLE = sys.executable

logger = logging.getLogger(__name__)


def create_single_metric_card(parent: tk.Frame, metric_name: str, value: Any, unit: str = '', 
                               description: str = '', progress: Optional[float] = None,
                               title: str = '', subtitle: str = '') -> tk.Frame:
    """
    Create a modern single metric card for grid layout with professional titles.
    
    Args:
        parent: Parent frame (grid container)
        metric_name: Name of the metric (displayed as subtitle if title is provided)
        value: Metric value
        unit: Unit string
        description: Description text
        progress: Progress value (0.0-1.0) for progress bar
        title: Main card title (if provided, replaces category)
        subtitle: Card subtitle (if not provided, uses metric_name)
        
    Returns:
        Card frame
    """
    from src.ui.ui_constants import (
        BG_COLOR_CARD, BG_COLOR_ENTRY, BG_COLOR_HOVER, FG_COLOR_WHITE, FG_COLOR_TEXT,
        FG_COLOR_SECONDARY, SUCCESS_COLOR, FONT_FAMILY, FONT_SIZE_NORMAL,
        FONT_SIZE_HEADING, BORDER_COLOR, ACCENT_COLOR, BUTTON_SECONDARY
    )
    
    # Card container with modern styling
    card = tk.Frame(
        parent,
        bg=BG_COLOR_CARD,
        relief=tk.FLAT,
        bd=0,
        highlightbackground=BORDER_COLOR,
        highlightthickness=1
    )
    
    # Inner padding frame for better spacing
    inner_frame = tk.Frame(card, bg=BG_COLOR_CARD)
    inner_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    # Title (if provided) - main card title
    if title:
        title_label = tk.Label(
            inner_frame,
            text=title,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            fg=FG_COLOR_WHITE,
            bg=BG_COLOR_CARD,
            anchor='w'
        )
        title_label.pack(fill=tk.X, pady=(0, 4))
    
    # Subtitle or metric name
    subtitle_text = subtitle if subtitle else metric_name
    subtitle_label = tk.Label(
        inner_frame,
        text=subtitle_text,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL - 1),
        fg=FG_COLOR_SECONDARY,
        bg=BG_COLOR_CARD,
        anchor='w',
        wraplength=200
    )
    subtitle_label.pack(fill=tk.X, pady=(0, 8))
    
    # Value display
    value_frame = tk.Frame(inner_frame, bg=BG_COLOR_CARD)
    value_frame.pack(fill=tk.X, pady=(0, 8))
    
    # Determine value color based on metric type and value
    if isinstance(value, (int, float)):
        if progress is not None and 0 <= progress <= 1:
            # For 0-1 metrics, green if > 0.5, blue otherwise
            value_color = SUCCESS_COLOR if progress > 0.5 else ACCENT_COLOR
        elif 'Time' in metric_name or 'time' in metric_name.lower():
            # For time metrics, lower is better (green for low, red-ish for high)
            value_color = SUCCESS_COLOR if value < 100 else FG_COLOR_WHITE
        else:
            # Default: green for good values
            value_color = SUCCESS_COLOR if value > 0.5 else FG_COLOR_WHITE
        
        # Format value based on type
        if abs(value) < 0.01:
            value_text = f"{value:.6f}"
        elif abs(value) < 1:
            value_text = f"{value:.4f}"
        elif abs(value) < 1000:
            value_text = f"{value:.2f}"
        else:
            value_text = f"{value:,.0f}"
    else:
        value_text = str(value)
        value_color = FG_COLOR_WHITE
    
    # Large value display
    value_label = tk.Label(
        value_frame,
        text=value_text,
        font=(FONT_FAMILY, FONT_SIZE_HEADING + 2, 'bold'),
        fg=value_color,
        bg=BG_COLOR_CARD
    )
    value_label.pack(side=tk.LEFT)
    
    # Unit label
    if unit:
        unit_label = tk.Label(
            value_frame,
            text=f" {unit}",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg=FG_COLOR_SECONDARY,
            bg=BG_COLOR_CARD
        )
        unit_label.pack(side=tk.LEFT, pady=(4, 0))
    
    # Progress bar for 0-1 metrics
    if progress is not None and isinstance(progress, (int, float)) and 0 <= progress <= 1:
        progress_container = tk.Frame(inner_frame, bg=BG_COLOR_CARD)
        progress_container.pack(fill=tk.X, pady=(8, 0))
        
        # Progress bar background
        progress_bg = tk.Frame(
            progress_container,
            bg=BG_COLOR_ENTRY,
            height=8,
            relief=tk.FLAT
        )
        progress_bg.pack(fill=tk.X)
        progress_bg.pack_propagate(False)
        
        # Progress fill
        progress_width = int(progress * 250)  # Max width for card
        if progress_width > 0:
            progress_fill = tk.Frame(
                progress_bg,
                bg=SUCCESS_COLOR if progress > 0.5 else ACCENT_COLOR,
                width=progress_width,
                height=8
            )
            progress_fill.pack(side=tk.LEFT, fill=tk.Y)
        
        # Progress percentage text
        percent_text = f"{progress * 100:.1f}%"
        tk.Label(
            progress_container,
            text=percent_text,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL - 1),
            fg=FG_COLOR_SECONDARY,
            bg=BG_COLOR_CARD
        ).pack(pady=(4, 0))
    
    # Description
    if description:
        desc_label = tk.Label(
            inner_frame,
            text=description,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL - 1),
            fg=FG_COLOR_SECONDARY,
            bg=BG_COLOR_CARD,
            anchor='w',
            wraplength=220,
            justify=tk.LEFT
        )
        desc_label.pack(fill=tk.X, pady=(8, 0))
    
    return card


def create_section_header(parent: tk.Frame, title: str, icon: str) -> tk.Frame:
    """
    Create a section header for grouping cards.
    
    Args:
        parent: Parent frame
        title: Section title
        icon: Icon/emoji
        
    Returns:
        Header frame
    """
    from src.ui.ui_constants import (
        BG_COLOR_MAIN, FG_COLOR_WHITE, FONT_FAMILY, FONT_SIZE_HEADING
    )
    
    header = tk.Frame(parent, bg=BG_COLOR_MAIN)
    header.pack(fill=tk.X, pady=(20, 15), padx=10)
    
    tk.Label(
        header,
        text=f"{icon} {title}",
        font=(FONT_FAMILY, FONT_SIZE_HEADING + 1, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_MAIN
    ).pack(side=tk.LEFT)
    
    return header


def render_evaluation_results(parent: tk.Frame, results: Dict[str, Any]) -> None:
    """
    Render evaluation results in modern grid-based card UI.
    
    Args:
        parent: Parent frame to render into
        results: Evaluation results dictionary
    """
    from src.ui.ui_constants import BG_COLOR_MAIN
    
    # Clear existing widgets
    for widget in parent.winfo_children():
        widget.destroy()
    
    # Performance Metrics Section
    if 'performance_metrics' in results and 'average' in results['performance_metrics']:
        pm = results['performance_metrics']['average']
        
        # Create section header
        create_section_header(parent, "Performance Metrics", "📈")
        
        # Create grid container for performance metrics
        perf_grid = tk.Frame(parent, bg=BG_COLOR_MAIN)
        perf_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        row = 0
        col = 0
        max_cols = 3  # 3 columns for modern grid
        
        # Metric titles mapping for professional display
        metric_titles = {
            'Precision@5': ('Precision at Top 5', 'Measures accuracy of top 5 recommendations'),
            'Precision@10': ('Precision at Top 10', 'Measures accuracy of top 10 recommendations'),
            'Recall@5': ('Recall at Top 5', 'Measures coverage of relevant items in top 5'),
            'Recall@10': ('Recall at Top 10', 'Measures coverage of relevant items in top 10'),
            'NDCG@5': ('NDCG at Top 5', 'Normalized Discounted Cumulative Gain'),
            'NDCG@10': ('NDCG at Top 10', 'Normalized Discounted Cumulative Gain'),
            'MAP': ('Mean Average Precision', 'Average precision across all queries')
        }
        
        for key in ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10', 'MAP']:
            val = pm.get(key, 0)
            if isinstance(val, (int, float)):
                title, desc = metric_titles.get(key, (key, 'Higher is better'))
                card = create_single_metric_card(
                    perf_grid,
                    metric_name=key,
                    value=val,
                    unit='(0.0-1.0)',
                    description=desc,
                    progress=val,
                    title=title,
                    subtitle=key
                )
                card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
        
        # Configure grid weights
        for c in range(max_cols):
            perf_grid.columnconfigure(c, weight=1, uniform='metric_col')
    
    # Baseline Comparison Section - Show summary per method
    if 'baseline_comparison' in results:
        bc = results['baseline_comparison']
        if 'error' not in bc:
            create_section_header(parent, "Baseline Comparison", "⚖️")
            
            baseline_grid = tk.Frame(parent, bg=BG_COLOR_MAIN)
            baseline_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            row = 0
            col = 0
            max_cols = 3
            
            # Method display names
            method_names = {
                'Random': 'Random Baseline',
                'TF-IDF Only': 'TF-IDF Baseline',
                'BM25 Only': 'BM25 Baseline'
            }
            
            for method, method_metrics in bc.items():
                if isinstance(method_metrics, dict):
                    # Get key metrics for summary
                    precision_10 = method_metrics.get('Precision@10', 0)
                    ndcg_10 = method_metrics.get('NDCG@10', 0)
                    map_score = method_metrics.get('MAP', 0)
                    
                    # Use average of key metrics as main value
                    avg_score = (precision_10 + ndcg_10 + map_score) / 3 if (precision_10 + ndcg_10 + map_score) > 0 else 0
                    
                    # Create summary card per method
                    method_display = method_names.get(method, method)
                    card = create_single_metric_card(
                        baseline_grid,
                        metric_name=method,
                        value=avg_score,
                        unit='(0.0-1.0)',
                        description=f'P@10: {precision_10:.3f} | NDCG@10: {ndcg_10:.3f} | MAP: {map_score:.3f}',
                        progress=avg_score,
                        title=method_display,
                        subtitle='Average Performance Score'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            
            # Configure grid weights
            for c in range(max_cols):
                baseline_grid.columnconfigure(c, weight=1, uniform='metric_col')
    
    # Diversity, Novelty, Coverage Section
    if 'diversity_novelty_coverage' in results:
        dnc = results['diversity_novelty_coverage']
        if 'error' not in dnc:
            create_section_header(parent, "Diversity, Novelty & Coverage", "🎯")
            
            dnc_grid = tk.Frame(parent, bg=BG_COLOR_MAIN)
            dnc_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            row = 0
            col = 0
            max_cols = 3
            
            # Title mapping for diversity metrics
            diversity_titles = {
                'Catalog Coverage': ('Catalog Coverage', 'Percentage of products recommended'),
                'Intra-List Diversity': ('Recommendation Diversity', 'Variety within recommendations'),
                'Novelty Score': ('Novelty Score', 'How novel recommendations are')
            }
            
            for metric_name, value in dnc.items():
                if isinstance(value, (int, float)):
                    if 'Coverage' in metric_name:
                        title, desc = diversity_titles.get(metric_name, ('Catalog Coverage', 'Percentage of catalog'))
                        progress = value
                    elif 'Diversity' in metric_name:
                        title, desc = diversity_titles.get(metric_name, ('Recommendation Diversity', 'Higher = more diverse'))
                        progress = value
                    elif 'Novelty' in metric_name:
                        title, desc = diversity_titles.get(metric_name, ('Novelty Score', 'Higher = more novel'))
                        progress = value
                    else:
                        title = metric_name.replace('_', ' ').title()
                        desc = ''
                        progress = value if 0 <= value <= 1 else None
                    
                    card = create_single_metric_card(
                        dnc_grid,
                        metric_name=metric_name,
                        value=value,
                        unit='(0.0-1.0)',
                        description=desc,
                        progress=progress,
                        title=title,
                        subtitle=metric_name
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            
            # Configure grid weights
            for c in range(max_cols):
                dnc_grid.columnconfigure(c, weight=1, uniform='metric_col')
    
    # Scalability & Efficiency Section
    if 'scalability_efficiency' in results:
        se = results['scalability_efficiency']
        if 'error' not in se:
            create_section_header(parent, "Scalability & Efficiency", "⚡")
            
            scale_grid = tk.Frame(parent, bg=BG_COLOR_MAIN)
            scale_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            row = 0
            col = 0
            max_cols = 3
            
            if 'query_performance' in se:
                qp = se['query_performance']
                mean_time = qp.get('mean_time_ms', 0)
                if isinstance(mean_time, (int, float)):
                    card = create_single_metric_card(
                        scale_grid,
                        metric_name='Mean Query Time',
                        value=mean_time,
                        unit='ms',
                        description='Average response time per query',
                        title='Mean Query Time',
                        subtitle='Average Response Time'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                if 'median_time_ms' in qp:
                    median_time = qp.get('median_time_ms', 0)
                    if isinstance(median_time, (int, float)):
                        card = create_single_metric_card(
                            scale_grid,
                            metric_name='Median Query Time',
                            value=median_time,
                            unit='ms',
                            description='Median response time per query',
                            title='Median Query Time',
                            subtitle='Median Response Time'
                        )
                        card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
            
            if 'memory_usage' in se:
                mu = se['memory_usage']
                rss_mb = mu.get('rss_mb', 0)
                if isinstance(rss_mb, (int, float)) and rss_mb > 0:
                    card = create_single_metric_card(
                        scale_grid,
                        metric_name='Memory Usage (RSS)',
                        value=rss_mb,
                        unit='MB',
                        description='Resident Set Size memory usage',
                        title='Memory Usage (RSS)',
                        subtitle='System Memory Consumption'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                if 'percent' in mu:
                    mem_percent = mu.get('percent', 0)
                    if isinstance(mem_percent, (int, float)) and mem_percent > 0:
                        card = create_single_metric_card(
                            scale_grid,
                            metric_name='Memory Usage',
                            value=mem_percent,
                            unit='%',
                            description='Percentage of system memory',
                            title='Memory Usage',
                            subtitle='System Memory Percentage'
                        )
                        card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
            
            if 'dataset_stats' in se:
                ds = se['dataset_stats']
                if 'num_products' in ds:
                    card = create_single_metric_card(
                        scale_grid,
                        metric_name='Dataset Size',
                        value=ds.get('num_products', 0),
                        unit='products',
                        description='Total products in dataset',
                        title='Dataset Size',
                        subtitle='Total Products'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            
            # Configure grid weights
            for c in range(max_cols):
                scale_grid.columnconfigure(c, weight=1, uniform='metric_col')
    
    # Cold Start Section
    if 'cold_start' in results:
        cs = results['cold_start']
        if 'error' not in cs:
            create_section_header(parent, "Cold Start Handling", "❄️")
            
            cold_grid = tk.Frame(parent, bg=BG_COLOR_MAIN)
            cold_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            row = 0
            col = 0
            max_cols = 3
            
            # Cold start metric titles
            cold_start_titles = {
                'new_user_avg_recommendations': ('New User Recommendations', 'Average recommendations for new users'),
                'new_item_avg_similar': ('New Item Similarity', 'Average similarity for new items')
            }
            
            for key, value in cs.items():
                if isinstance(value, (int, float)):
                    title, desc = cold_start_titles.get(key, (key.replace('_', ' ').title(), ''))
                    unit = '(average count)' if 'avg_recommendations' in key or 'avg_similar' in key else ''
                    card = create_single_metric_card(
                        cold_grid,
                        metric_name=key,
                        value=value,
                        unit=unit,
                        description=desc,
                        title=title,
                        subtitle=key.replace('_', ' ').title()
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            
            # Configure grid weights
            for c in range(max_cols):
                cold_grid.columnconfigure(c, weight=1, uniform='metric_col')
    
    # Parameter Tuning Section
    if 'parameter_tuning' in results:
        pt = results['parameter_tuning']
        if 'error' not in pt:
            if 'hybrid_weights' in pt:
                hw = pt['hybrid_weights']
                if hw:
                    try:
                        best_weight = max(hw.items(), key=lambda x: x[1].get('NDCG@10', 0) if isinstance(x[1], dict) else 0)
                        bm25_w = best_weight[0]
                        semantic_w = 1.0 - bm25_w if isinstance(bm25_w, (int, float)) else 0.0
                        best_ndcg = best_weight[1].get('NDCG@10', 0) if isinstance(best_weight[1], dict) else 0
                        
                        create_section_header(parent, "Parameter Tuning Results", "🎛️")
                        
                        tune_grid = tk.Frame(parent, bg=BG_COLOR_MAIN)
                        tune_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
                        
                        row = 0
                        col = 0
                        max_cols = 3
                        
                        if isinstance(bm25_w, (int, float)):
                            card = create_single_metric_card(
                                tune_grid,
                                metric_name='Best BM25 Weight',
                                value=bm25_w,
                                unit='(0.0-1.0)',
                                description='Optimal BM25 weight in hybrid search',
                                title='Best BM25 Weight',
                                subtitle='Optimal Hybrid Parameter'
                            )
                            card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                            col += 1
                            if col >= max_cols:
                                col = 0
                                row += 1
                            
                            card = create_single_metric_card(
                                tune_grid,
                                metric_name='Best Semantic Weight',
                                value=semantic_w,
                                unit='(0.0-1.0)',
                                description='Optimal semantic embedding weight',
                                title='Best Semantic Weight',
                                subtitle='Optimal Hybrid Parameter'
                            )
                            card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                            col += 1
                            if col >= max_cols:
                                col = 0
                                row += 1
                        
                        if isinstance(best_ndcg, (int, float)):
                            card = create_single_metric_card(
                                tune_grid,
                                metric_name='Best NDCG@10',
                                value=best_ndcg,
                                unit='(0.0-1.0)',
                                description='Best performance with tuned parameters',
                                progress=best_ndcg,
                                title='Best NDCG@10',
                                subtitle='Optimal Performance Score'
                            )
                            card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                            col += 1
                            if col >= max_cols:
                                col = 0
                                row += 1
                        
                        # Configure grid weights
                        for c in range(max_cols):
                            tune_grid.columnconfigure(c, weight=1, uniform='metric_col')
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error formatting parameter tuning: {e}")
    
    # A/B Testing Section
    if 'ab_testing' in results:
        ab = results['ab_testing']
        if 'error' not in ab:
            create_section_header(parent, "A/B Testing Results", "🧪")
            
            ab_grid = tk.Frame(parent, bg=BG_COLOR_MAIN)
            ab_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            row = 0
            col = 0
            max_cols = 3
            
            if 'control' in ab and 'treatment' in ab:
                control = ab['control']
                treatment = ab['treatment']
                
                ctrl_ctr = control.get('mean_ctr', 0)
                treat_ctr = treatment.get('mean_ctr', 0)
                
                if isinstance(ctrl_ctr, (int, float)):
                    card = create_single_metric_card(
                        ab_grid,
                        metric_name='Control Group CTR',
                        value=ctrl_ctr,
                        unit='(0.0-1.0)',
                        description='Baseline click-through rate',
                        progress=ctrl_ctr,
                        title='Control Group CTR',
                        subtitle='Baseline Performance'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                if isinstance(treat_ctr, (int, float)):
                    card = create_single_metric_card(
                        ab_grid,
                        metric_name='Treatment Group CTR',
                        value=treat_ctr,
                        unit='(0.0-1.0)',
                        description='Experimental click-through rate',
                        progress=treat_ctr,
                        title='Treatment Group CTR',
                        subtitle='Experimental Performance'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                ctrl_rating = control.get('mean_rating', 0)
                treat_rating = treatment.get('mean_rating', 0)
                if isinstance(ctrl_rating, (int, float)) and ctrl_rating > 0:
                    card = create_single_metric_card(
                        ab_grid,
                        metric_name='Control Group Rating',
                        value=ctrl_rating,
                        unit='/ 5.0 stars',
                        description='Baseline user satisfaction rating',
                        title='Control Group Rating',
                        subtitle='Baseline Satisfaction'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                if isinstance(treat_rating, (int, float)) and treat_rating > 0:
                    card = create_single_metric_card(
                        ab_grid,
                        metric_name='Treatment Group Rating',
                        value=treat_rating,
                        unit='/ 5.0 stars',
                        description='Experimental user satisfaction rating',
                        title='Treatment Group Rating',
                        subtitle='Experimental Satisfaction'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            
            if 'ctr_test' in ab and 'error' not in ab['ctr_test']:
                ctr = ab['ctr_test']
                improvement = ctr.get('improvement', 0)
                if isinstance(improvement, (int, float)):
                    card = create_single_metric_card(
                        ab_grid,
                        metric_name='CTR Improvement',
                        value=improvement,
                        unit='%',
                        description='Percentage improvement over baseline',
                        title='CTR Improvement',
                        subtitle='Performance Gain'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                p_value = ctr.get('p_value', 0)
                if isinstance(p_value, (int, float)):
                    card = create_single_metric_card(
                        ab_grid,
                        metric_name='P-value',
                        value=p_value,
                        description='Statistical significance test result',
                        title='P-value',
                        subtitle='Statistical Significance'
                    )
                    card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                significant = ctr.get('significant', False)
                card = create_single_metric_card(
                    ab_grid,
                    metric_name='Statistically Significant',
                    value='Yes' if significant else 'No',
                    description='Result is statistically significant (p < 0.05)',
                    title='Statistical Significance',
                    subtitle='Test Result'
                )
                card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
            
            # Configure grid weights
            for c in range(max_cols):
                ab_grid.columnconfigure(c, weight=1, uniform='metric_col')
    
    # Empty state
    if not parent.winfo_children():
        from src.ui.ui_constants import BG_COLOR_MAIN, FG_COLOR_SECONDARY, FONT_FAMILY, FONT_SIZE_HEADING, FONT_SIZE_NORMAL
        
        empty_frame = tk.Frame(parent, bg=BG_COLOR_MAIN)
        empty_frame.pack(fill=tk.BOTH, expand=True, pady=50)
        
        tk.Label(
            empty_frame,
            text="📊",
            font=(FONT_FAMILY, 48),
            fg=FG_COLOR_SECONDARY,
            bg=BG_COLOR_MAIN
        ).pack(pady=20)
        
        tk.Label(
            empty_frame,
            text="No evaluation results available",
            font=(FONT_FAMILY, FONT_SIZE_HEADING),
            fg=FG_COLOR_SECONDARY,
            bg=BG_COLOR_MAIN
        ).pack(pady=10)
        
        tk.Label(
            empty_frame,
            text="Click 'Run Quick Evaluation' to generate metrics",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg=FG_COLOR_SECONDARY,
            bg=BG_COLOR_MAIN
        ).pack()


def format_metrics_display(results: Dict[str, Any]) -> str:
    """
    Format evaluation results for display in the UI (fallback text format).
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Formatted string for display
    """
    output = []
    output.append("=" * 70)
    output.append("EVALUATION RESULTS")
    output.append("=" * 70)
    output.append("")
    
    # Performance Metrics
    if 'performance_metrics' in results and 'average' in results['performance_metrics']:
        pm = results['performance_metrics']['average']
        output.append("PERFORMANCE METRICS")
        output.append("-" * 70)
        output.append(f"Precision@5:  {pm.get('Precision@5', 0):.4f} (0.0-1.0, higher is better)")
        output.append(f"Precision@10: {pm.get('Precision@10', 0):.4f} (0.0-1.0, higher is better)")
        output.append(f"Recall@5:     {pm.get('Recall@5', 0):.4f} (0.0-1.0, higher is better)")
        output.append(f"Recall@10:    {pm.get('Recall@10', 0):.4f} (0.0-1.0, higher is better)")
        output.append(f"NDCG@5:       {pm.get('NDCG@5', 0):.4f} (0.0-1.0, higher is better)")
        output.append(f"NDCG@10:      {pm.get('NDCG@10', 0):.4f} (0.0-1.0, higher is better)")
        output.append(f"MAP:          {pm.get('MAP', 0):.4f} (0.0-1.0, higher is better)")
        # Note about RMSE/MAE
        output.append("")
        output.append("Note: RMSE and MAE are for rating prediction tasks.")
        output.append("      Content-based filtering uses similarity scores, not ratings.")
        output.append("      These metrics are more relevant for collaborative filtering.")
        output.append("")
    
    # Baseline Comparison
    if 'baseline_comparison' in results:
        bc = results['baseline_comparison']
        if 'error' not in bc:
            output.append("BASELINE COMPARISON")
            output.append("-" * 70)
            for method, metrics in bc.items():
                if isinstance(metrics, dict):
                    output.append(f"\n{method}:")
                    if 'Precision@10' in metrics:
                        val = metrics.get('Precision@10', 0)
                        if isinstance(val, (int, float)):
                            output.append(f"  Precision@10: {val:.4f} (0.0-1.0, higher is better)")
                        else:
                            output.append(f"  Precision@10: {val}")
                    if 'NDCG@10' in metrics:
                        val = metrics.get('NDCG@10', 0)
                        if isinstance(val, (int, float)):
                            output.append(f"  NDCG@10:      {val:.4f} (0.0-1.0, higher is better)")
                        else:
                            output.append(f"  NDCG@10:      {val}")
                    if 'MAP' in metrics:
                        val = metrics.get('MAP', 0)
                        if isinstance(val, (int, float)):
                            output.append(f"  MAP:          {val:.4f} (0.0-1.0, higher is better)")
                        else:
                            output.append(f"  MAP:          {val}")
            output.append("")
    
    # Diversity, Novelty, Coverage
    if 'diversity_novelty_coverage' in results:
        dnc = results['diversity_novelty_coverage']
        if 'error' not in dnc:
            output.append("DIVERSITY, NOVELTY & COVERAGE")
            output.append("-" * 70)
            for metric, value in dnc.items():
                # Handle both numeric and string values
                if isinstance(value, (int, float)):
                    # Add appropriate units based on metric name
                    if 'Coverage' in metric:
                        output.append(f"{metric}: {value:.4f} (0.0-1.0, percentage of catalog)")
                    elif 'Diversity' in metric or 'Novelty' in metric:
                        output.append(f"{metric}: {value:.4f} (0.0-1.0, higher = more diverse/novel)")
                    else:
                        output.append(f"{metric}: {value:.4f}")
                else:
                    output.append(f"{metric}: {value}")
            output.append("")
    
    # Scalability
    if 'scalability_efficiency' in results:
        se = results['scalability_efficiency']
        if 'error' not in se:
            output.append("SCALABILITY & EFFICIENCY")
            output.append("-" * 70)
            if 'query_performance' in se:
                qp = se['query_performance']
                mean_time = qp.get('mean_time_ms', 0)
                if isinstance(mean_time, (int, float)):
                    output.append(f"Mean Query Time: {mean_time:.2f} ms (milliseconds, lower is better)")
                if 'median_time_ms' in qp:
                    median_time = qp.get('median_time_ms', 0)
                    if isinstance(median_time, (int, float)):
                        output.append(f"Median Query Time: {median_time:.2f} ms")
                if 'std_time_ms' in qp:
                    std_time = qp.get('std_time_ms', 0)
                    if isinstance(std_time, (int, float)):
                        output.append(f"Std Dev Query Time: {std_time:.2f} ms")
            if 'memory_usage' in se:
                mu = se['memory_usage']
                rss_mb = mu.get('rss_mb', 0)
                if isinstance(rss_mb, (int, float)) and rss_mb > 0:
                    output.append(f"Memory Usage (RSS): {rss_mb:.2f} MB (megabytes)")
                if 'percent' in mu:
                    mem_percent = mu.get('percent', 0)
                    if isinstance(mem_percent, (int, float)) and mem_percent > 0:
                        output.append(f"Memory Usage: {mem_percent:.2f}% (percentage of system memory)")
            if 'dataset_stats' in se:
                ds = se['dataset_stats']
                if 'num_products' in ds:
                    output.append(f"Dataset Size: {ds.get('num_products', 0):,} products")
            output.append("")
    
    # Cold Start
    if 'cold_start' in results:
        cs = results['cold_start']
        if 'error' not in cs:
            output.append("COLD START HANDLING")
            output.append("-" * 70)
            for key, value in cs.items():
                if isinstance(value, (int, float)):
                    if 'avg_recommendations' in key or 'avg_similar' in key:
                        output.append(f"{key}: {value:.2f} (average count)")
                    else:
                        output.append(f"{key}: {value}")
                else:
                    output.append(f"{key}: {value}")
            output.append("")
    
    # Parameter Tuning
    if 'parameter_tuning' in results:
        pt = results['parameter_tuning']
        if 'error' not in pt:
            output.append("PARAMETER TUNING RESULTS")
            output.append("-" * 70)
            if 'hybrid_weights' in pt:
                hw = pt['hybrid_weights']
                if hw:
                    # Find best weight
                    try:
                        best_weight = max(hw.items(), key=lambda x: x[1].get('NDCG@10', 0) if isinstance(x[1], dict) else 0)
                        bm25_w = best_weight[0]
                        semantic_w = 1.0 - bm25_w if isinstance(bm25_w, (int, float)) else 0.0
                        best_ndcg = best_weight[1].get('NDCG@10', 0) if isinstance(best_weight[1], dict) else 0
                        if isinstance(bm25_w, (int, float)):
                            output.append(f"Best Hybrid Weight - BM25: {bm25_w:.2f} (0.0-1.0)")
                            output.append(f"Best Hybrid Weight - Semantic: {semantic_w:.2f} (0.0-1.0)")
                        if isinstance(best_ndcg, (int, float)):
                            output.append(f"  Best NDCG@10: {best_ndcg:.4f} (0.0-1.0, higher is better)")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error formatting parameter tuning: {e}")
            output.append("")
    
    # A/B Testing
    if 'ab_testing' in results:
        ab = results['ab_testing']
        if 'error' not in ab:
            output.append("A/B TESTING RESULTS")
            output.append("-" * 70)
            if 'control' in ab and 'treatment' in ab:
                control = ab['control']
                treatment = ab['treatment']
                ctrl_ctr = control.get('mean_ctr', 0)
                treat_ctr = treatment.get('mean_ctr', 0)
                if isinstance(ctrl_ctr, (int, float)):
                    output.append(f"Control Group CTR: {ctrl_ctr:.4f} (0.0-1.0, click-through rate)")
                if isinstance(treat_ctr, (int, float)):
                    output.append(f"Treatment Group CTR: {treat_ctr:.4f} (0.0-1.0, click-through rate)")
                ctrl_rating = control.get('mean_rating', 0)
                treat_rating = treatment.get('mean_rating', 0)
                if isinstance(ctrl_rating, (int, float)) and ctrl_rating > 0:
                    output.append(f"Control Group Rating: {ctrl_rating:.2f} / 5.0 (stars)")
                if isinstance(treat_rating, (int, float)) and treat_rating > 0:
                    output.append(f"Treatment Group Rating: {treat_rating:.2f} / 5.0 (stars)")
            if 'ctr_test' in ab and 'error' not in ab['ctr_test']:
                ctr = ab['ctr_test']
                improvement = ctr.get('improvement', 0)
                if isinstance(improvement, (int, float)):
                    output.append(f"CTR Improvement: {improvement:+.2f}% (percentage change)")
                p_value = ctr.get('p_value', 0)
                if isinstance(p_value, (int, float)):
                    output.append(f"P-value: {p_value:.4f} (< 0.05 indicates significance)")
                output.append(f"Statistically Significant: {'Yes' if ctr.get('significant', False) else 'No'} (p < 0.05)")
            output.append("")
    
    output.append("=" * 70)
    
    return "\n".join(output)


def load_latest_evaluation_results() -> Optional[Dict[str, Any]]:
    """
    Load the evaluation results JSON file.
    
    Returns:
        Dictionary with evaluation results or None if not found
    """
    # Get project root to find evaluation_results directory
    current_file = os.path.abspath(__file__)
    ui_dir = os.path.dirname(current_file)  # src/ui
    src_dir = os.path.dirname(ui_dir)        # src
    project_root = os.path.dirname(src_dir) # project_root
    
    eval_dir = os.path.join(project_root, "evaluation_results")
    results_file = os.path.join(eval_dir, "evaluation_results.json")
    
    if not os.path.exists(results_file):
        return None
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        return None


def run_quick_evaluation(callback=None):
    """
    Run quick evaluation in background thread.
    
    Args:
        callback: Optional callback function to call when evaluation completes
    """
    def run_eval():
        try:
            logger.info("Starting quick evaluation...")
            # Get project root directory (parent of src directory)
            # __file__ is at src/ui/evaluation_ui.py
            # Go up: src/ui -> src -> project_root
            current_file = os.path.abspath(__file__)
            ui_dir = os.path.dirname(current_file)  # src/ui
            src_dir = os.path.dirname(ui_dir)       # src
            project_root = os.path.dirname(src_dir) # project_root
            
            # Set environment variables
            env = os.environ.copy()
            env['TOKENIZERS_PARALLELISM'] = 'false'
            
            # Add project root to PYTHONPATH so src module can be found
            # Use absolute path and proper separator for the OS
            pythonpath = env.get('PYTHONPATH', '')
            project_root_abs = os.path.abspath(project_root)
            
            if pythonpath:
                # Use os.pathsep for cross-platform compatibility
                env['PYTHONPATH'] = f"{project_root_abs}{os.pathsep}{pythonpath}"
            else:
                env['PYTHONPATH'] = project_root_abs
            
            logger.debug(f"Running evaluation from: {project_root_abs}")
            logger.debug(f"PYTHONPATH set to: {env['PYTHONPATH']}")
            
            # Run evaluation script from project root as a module
            # Use sys.executable for cross-platform compatibility (works on Windows, macOS, Linux)
            result = subprocess.run(
                [PYTHON_EXECUTABLE, "-m", "src.evaluation.run_evaluation", 
                 "--quick", 
                 "--csv-path", "data/home appliance skus lowes.csv",
                 "--output-dir", "evaluation_results"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
                cwd=project_root_abs
            )
            
            if result.returncode == 0:
                logger.info("Quick evaluation completed successfully")
                if result.stdout:
                    logger.debug(f"Evaluation output: {result.stdout[:500]}")
                if callback:
                    callback(True, "Evaluation completed successfully!")
            else:
                error_msg = result.stderr[:500] if result.stderr else result.stdout[:500] if result.stdout else "Unknown error"
                logger.error(f"Evaluation failed (return code {result.returncode}): {error_msg}")
                if callback:
                    callback(False, f"Evaluation failed: {error_msg}")
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out")
            if callback:
                callback(False, "Evaluation timed out (took longer than 5 minutes)")
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            if callback:
                callback(False, f"Error: {str(e)}")
    
    thread = threading.Thread(target=run_eval, daemon=True)
    thread.start()


def open_evaluation_report():
    """Open the evaluation report file."""
    # Get project root to find evaluation_results directory
    current_file = os.path.abspath(__file__)
    ui_dir = os.path.dirname(current_file)  # src/ui
    src_dir = os.path.dirname(ui_dir)        # src
    project_root = os.path.dirname(src_dir) # project_root
    
    eval_dir = os.path.join(project_root, "evaluation_results")
    report_file = os.path.join(eval_dir, "evaluation_report.txt")
    
    if not os.path.exists(report_file):
        return None
    
    try:
        # Open file with default text editor
        if os.name == 'nt':  # Windows
            os.startfile(report_file)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', report_file])
        return report_file
    except Exception as e:
        logger.error(f"Error opening report: {e}")
        return None


def open_evaluation_folder():
    """Open the evaluation results folder."""
    # Get project root to find evaluation_results directory
    current_file = os.path.abspath(__file__)
    ui_dir = os.path.dirname(current_file)  # src/ui
    src_dir = os.path.dirname(ui_dir)        # src
    project_root = os.path.dirname(src_dir) # project_root
    
    eval_dir = os.path.join(project_root, "evaluation_results")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
    
    try:
        # Open folder with default file manager
        if os.name == 'nt':  # Windows
            os.startfile(eval_dir)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', eval_dir])
    except Exception as e:
        logger.error(f"Error opening folder: {e}")
