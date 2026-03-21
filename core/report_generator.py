"""
评测报告生成器
生成 Markdown 和 HTML 格式的评测报告
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """评测报告生成器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.template_dir = Path(__file__).parent / "templates"

    def generate(self, results: Dict[str, Any], output_path: str):
        """生成报告"""
        output_path = Path(output_path)

        # 生成 Markdown 报告
        md_path = output_path.with_suffix('.md')
        self._generate_markdown(results, md_path)

        # 生成 HTML 报告
        html_path = output_path.with_suffix('.html')
        self._generate_html(results, html_path)

        # 生成 JSON 报告（原始数据）
        json_path = output_path.with_suffix('.json')
        self._generate_json(results, json_path)

        return {
            'markdown': str(md_path),
            'html': str(html_path),
            'json': str(json_path)
        }

    def _generate_markdown(self, results: Dict, output_path: Path):
        """生成 Markdown 报告"""
        lines = []

        # 标题
        lines.append("# MiMo-V2-Pro 能力评测报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 综合得分
        lines.append("## 综合得分")
        lines.append("")

        final_scores = results.get('final_scores', {})
        for model_id, scores in final_scores.items():
            lines.append(f"### {model_id}")
            lines.append("")
            lines.append(f"**总体得分**: {scores.get('overall_score', 0):.2f}")
            lines.append("")

            # 维度分解
            lines.append("**维度得分**:")
            lines.append("")
            lines.append("| 维度 | 原始得分 | 加权贡献 |")
            lines.append("|------|----------|----------|")

            breakdown = scores.get('dimension_breakdown', {})
            for dim, dim_scores in breakdown.items():
                raw = dim_scores.get('raw_score', 0)
                weighted = dim_scores.get('weighted_contribution', 0)
                lines.append(f"| {dim} | {raw:.2f} | {weighted:.2f} |")

            lines.append("")

        # 详细结果
        lines.append("## 详细评测结果")
        lines.append("")

        detailed = results.get('detailed_results', {})
        for model_id, dimensions in detailed.items():
            lines.append(f"### {model_id}")
            lines.append("")

            for dim_name, dim_result in dimensions.items():
                lines.append(f"#### {dim_name}")
                lines.append("")

                # 基准测试结果 - 处理 dataclass 或 dict
                if hasattr(dim_result, 'benchmark_results'):
                    # 是 DimensionScore 对象
                    bench_results = dim_result.benchmark_results
                else:
                    # 是字典
                    bench_results = dim_result.get('benchmark_results', [])

                for bench in bench_results:
                    if hasattr(bench, 'benchmark'):
                        bench_name = bench.benchmark
                        score = bench.score
                        correct = bench.correct_samples
                        total = bench.total_samples
                    else:
                        bench_name = bench.get('benchmark', 'unknown')
                        score = bench.get('score', 0)
                        correct = bench.get('correct_samples', 0)
                        total = bench.get('total_samples', 0)

                    lines.append(f"**{bench_name}**: {score:.2%} ({correct}/{total})")
                    lines.append("")

        # 配置信息
        lines.append("## 评测配置")
        lines.append("")
        lines.append("```yaml")
        lines.append(json.dumps(self.config, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        # 写入文件
        output_path.write_text('\n'.join(lines), encoding='utf-8')

    def _generate_html(self, results: Dict, output_path: Path):
        """生成 HTML 报告"""
        final_scores = results.get('final_scores', {})

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MiMo-V2-Pro 评测报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #ff6900;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #333;
            margin: 30px 0 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ddd;
        }}
        h3 {{
            color: #555;
            margin: 20px 0 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .score-card {{
            display: inline-block;
            background: linear-gradient(135deg, #ff6900, #ff8533);
            color: white;
            padding: 20px 40px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .score-card .score {{
            font-size: 48px;
            font-weight: bold;
        }}
        .score-card .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #ff6900;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f8f8;
        }}
        .dimension-score {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .dimension-score .name {{
            width: 150px;
            font-weight: 500;
        }}
        .dimension-score .bar {{
            flex: 1;
            height: 24px;
            background: #eee;
            border-radius: 12px;
            overflow: hidden;
        }}
        .dimension-score .fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff6900, #ff8533);
            border-radius: 12px;
            transition: width 0.3s ease;
        }}
        .dimension-score .value {{
            width: 60px;
            text-align: right;
            font-weight: 600;
            color: #ff6900;
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .model-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>MiMo-V2-Pro 能力评测报告</h1>
    <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
"""

        # 综合得分部分
        html += '    <h2>综合得分</h2>\n'

        for model_id, scores in final_scores.items():
            overall = scores.get('overall_score', 0)
            breakdown = scores.get('dimension_breakdown', {})

            html += f'''
    <div class="model-section">
        <h3>{model_id}</h3>
        <div class="score-card">
            <div class="score">{overall:.1f}</div>
            <div class="label">总体得分</div>
        </div>

        <h4>维度得分</h4>
'''
            for dim, dim_scores in breakdown.items():
                raw = dim_scores.get('raw_score', 0)
                html += f'''
        <div class="dimension-score">
            <div class="name">{dim}</div>
            <div class="bar">
                <div class="fill" style="width: {raw}%"></div>
            </div>
            <div class="value">{raw:.1f}</div>
        </div>
'''

            html += '    </div>\n'

        # 详细结果
        html += '    <h2>详细评测结果</h2>\n'

        detailed = results.get('detailed_results', {})
        for model_id, dimensions in detailed.items():
            html += f'    <div class="model-section">\n'
            html += f'        <h3>{model_id}</h3>\n'

            for dim_name, dim_result in dimensions.items():
                html += f'        <h4>{dim_name}</h4>\n'
                html += '''
        <table>
            <thead>
                <tr>
                    <th>评测基准</th>
                    <th>得分</th>
                    <th>正确数/总数</th>
                    <th>平均延迟</th>
                </tr>
            </thead>
            <tbody>
'''
                # 处理 dataclass 或 dict
                if hasattr(dim_result, 'benchmark_results'):
                    bench_results = dim_result.benchmark_results
                else:
                    bench_results = dim_result.get('benchmark_results', [])

                for bench in bench_results:
                    if hasattr(bench, 'benchmark'):
                        bench_name = bench.benchmark
                        score = bench.score
                        correct = bench.correct_samples
                        total = bench.total_samples
                        latency = bench.latency_avg
                    else:
                        bench_name = bench.get('benchmark', 'unknown')
                        score = bench.get('score', 0)
                        correct = bench.get('correct_samples', 0)
                        total = bench.get('total_samples', 0)
                        latency = bench.get('latency_avg', 0)

                    html += f'''                <tr>
                    <td>{bench_name}</td>
                    <td>{score:.2%}</td>
                    <td>{correct}/{total}</td>
                    <td>{latency:.2f}s</td>
                </tr>
'''

                html += '''            </tbody>
        </table>
'''

            html += '    </div>\n'

        # 结束
        html += '''
</body>
</html>
'''

        output_path.write_text(html, encoding='utf-8')

    def _generate_json(self, results: Dict, output_path: Path):
        """生成 JSON 报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def generate_comparison(self, results_list: List[Dict], output_path: str):
        """生成多模型对比报告"""
        output_path = Path(output_path)

        lines = []
        lines.append("# 多模型对比评测报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 提取所有模型
        models = []
        for result in results_list:
            final_scores = result.get('final_scores', {})
            models.extend(final_scores.keys())

        models = list(set(models))

        # 对比表格
        lines.append("## 综合得分对比")
        lines.append("")

        headers = ["模型", "总体得分"] + [dim for dim in self.config.get('dimensions', {}).keys()]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for result in results_list:
            final_scores = result.get('final_scores', {})
            for model_id, scores in final_scores.items():
                overall = scores.get('overall_score', 0)
                breakdown = scores.get('dimension_breakdown', {})

                row = [model_id, f"{overall:.2f}"]
                for dim in self.config.get('dimensions', {}).keys():
                    dim_score = breakdown.get(dim, {}).get('raw_score', 0)
                    row.append(f"{dim_score:.2f}")

                lines.append("| " + " | ".join(row) + " |")

        lines.append("")

        # 写入
        output_path = Path(output_path)
        output_path.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
