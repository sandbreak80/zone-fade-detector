#!/usr/bin/env python3
"""
Create HTML index for Seaborn visualizations
"""

from pathlib import Path

def create_index_html():
    """Create an index.html file to display all visualizations."""
    output_dir = Path("outputs/visualizations")
    png_files = list(output_dir.glob("*.png"))
    png_files.sort()
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Seaborn Zone Fade Visualizations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .chart-item {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-item h3 {
            margin-top: 0;
            color: #1f77b4;
        }
        .chart-item img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .chart-item a {
            color: #1f77b4;
            text-decoration: none;
        }
        .chart-item a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Seaborn Zone Fade Visualizations</h1>
        <div class="chart-grid">
"""
    
    for png_file in png_files:
        filename = png_file.name
        chart_name = filename.replace('_seaborn_chart.png', '').replace('_', ' ')
        
        html_content += f"""
            <div class="chart-item">
                <h3>{chart_name}</h3>
                <a href="{filename}" target="_blank">
                    <img src="{filename}" alt="{chart_name}">
                </a>
                <p><a href="{filename}" target="_blank">View Full Size</a></p>
            </div>
"""
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    # Write index.html
    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Created index.html with {len(png_files)} visualizations")
    print(f"📁 Location: {index_path}")
    print("🌐 Open this file in your browser to view all visualizations")
    return index_path

if __name__ == "__main__":
    create_index_html()