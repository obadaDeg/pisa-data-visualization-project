# PISA 2022 Data Analysis

This repository contains a comprehensive analysis of the Programme for International Student Assessment (PISA) 2022 dataset, exploring student performance patterns across countries, socioeconomic factors, and demographic characteristics.

## Project Overview

PISA is an international assessment coordinated by the OECD that evaluates education systems worldwide by testing the skills and knowledge of 15-year-old students. The 2022 assessment covered approximately 600,000 students across participating countries.

This project aims to uncover patterns and relationships in student performance, with particular attention to:

1. Country-level differences in academic achievement
2. Gender gaps across different subject domains
3. Impact of socioeconomic status on performance
4. Interplay between school climate and academic outcomes

## Repository Structure

```
└── obadadeg-pisa-data-visualization-project/
    ├── PISA Data Analysis.pptx           # Presentation summarizing key findings
    ├── data_analysis/                    # Data analysis code and outputs
    │   ├── data_visualization.ipynb      # Main analysis notebook
    │   ├── testing.ipynb                 # Experimental analysis notebook
    │   └── outputs/                      # Generated analysis outputs
    │       ├── project_summary.md        # Summary of key findings
    │       ├── cleaning/                 # Data cleaning reports and statistics
    │       ├── html/                     # HTML exports of notebooks
    │       ├── plots/                    # Generated visualizations
    │       ├── scripts/                  # Python scripts exported from notebooks
    │       └── tables/                   # CSV files with analysis results
    └── pisa_analysis_dashboard/          # Interactive dashboard application
        ├── README.md                     # Dashboard-specific documentation
        ├── public/                       # Static assets
        └── src/                          # Dashboard source code
```

## Key Research Questions

The analysis seeks to answer the following questions:

1. **Which countries demonstrate the highest performance across different subject areas in PISA 2022?**
   - Identifying top performers in mathematics, reading, and science
   - Analyzing regional patterns in educational achievement

2. **How do male and female students perform differently across the three core PISA domains?**
   - Quantifying gender gaps in mathematics, reading, and science
   - Identifying countries with the largest and smallest gender differences

3. **Which countries demonstrate the strongest link between socioeconomic status and academic performance?**
   - Measuring the correlation between ESCS (Economic, Social, and Cultural Status) and performance
   - Highlighting education systems that achieve both excellence and equity

4. **What is the relationship between a country's average socioeconomic status and its mathematics performance?**
   - Analyzing country-level associations between socioeconomic factors and outcomes
   - Identifying overperforming and underperforming countries relative to their resources

## Key Findings

### Country Performance Patterns

- East Asian education systems (Singapore, China, Japan, Korea) consistently lead in mathematics and science
- European countries like Estonia and Finland show strong balanced performance across all domains
- Significant variations exist even among countries with similar economic development levels

### Gender Differences

- Female students outperform male students in reading by approximately 30 points across most countries
- Male students maintain a small advantage (about 5 points) in mathematics
- Science performance shows minimal gender differences at the global level

### Socioeconomic Impact

- Countries like Hungary and Luxembourg show the strongest relationship between ESCS and performance
- Education systems like Estonia and Hong Kong achieve high performance with lower socioeconomic impact
- The strength of the ESCS-performance relationship varies significantly across countries

### School Climate Factors

- School discipline climate shows a positive relationship with performance
- The impact of school factors appears to be somewhat independent of socioeconomic status
- Student sense of belonging correlates positively with academic achievement

## Interactive Dashboard (Still in Development)

The repository includes an interactive dashboard built with React that visualizes key findings from the analysis. The dashboard offers:

- Country comparison views
- Gender gap visualizations
- Socioeconomic impact analysis
- Interactive filtering by region and performance level

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Node.js 14+ (for the dashboard)

### Setting Up the Analysis Environment

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/obadadeg-pisa-data-visualization-project.git
   cd obadadeg-pisa-data-visualization-project
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebook:
   ```
   jupyter notebook data_analysis/data_visualization.ipynb
   ```

### Running the Dashboard

1. Navigate to the dashboard directory:
   ```
   cd pisa_analysis_dashboard
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. Open your browser and visit `http://localhost:5173`

## Data Source

The PISA 2022 dataset was obtained from the OECD PISA database. The original dataset is available at:
https://www.oecd.org/pisa/data/

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OECD for providing the PISA 2022 dataset
- Contributors to the pandas, matplotlib, and seaborn Python libraries
- React and Vite communities for dashboard development tools
