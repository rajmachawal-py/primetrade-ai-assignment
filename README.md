# Primetrade.ai Data Science Task: Trader Performance vs. Market Sentiment

**Author:** Lakshay  
**Project Objective:** Analyze the correlation between Hyperliquid historical trader logs and Bitcoin's daily Fear & Greed sentiment index for the calendar year 2024 to uncover trading patterns and behavior.

---

## 📈 Key Findings & Insights

1. **Sentiment & Profitability**:
   * Trading performance is highly dependent on market sentiment. Statistical tests (**Kruskal-Wallis**, $p < 0.05$) show that Closed PnL distribution shifts significantly across different sentiment regimes.
   * Average trade PnL and win rates were highest during **Greed** (39.5% win rate) and **Extreme Greed** (47.5% win rate) zones.
   * Traders experienced overall net losses during **Fear** and **Extreme Fear** regimes.

2. **Trend Following vs. Contrarian Strategy**:
   * A simple **Trend Following** strategy (buying greed / shorting fear) returned **+36.09%** over 2024, outperforming the **Contrarian** strategy (-39.16%).
   * Since 2024 was a major bull market (Buy & Hold benchmark yielded **+119.25%**), contrarian shorting was heavily penalized.

3. **Trader Archetypes (K-Means Clustering)**:
   * **Momentum Followers** (25 accounts): Scale their trading volume and size in response to greed sentiment.
   * **Balanced Specialists** (3 accounts): Maintain consistent win rates and conservative trade sizing across all market conditions.
   * **Contrarian Whales** (3 accounts): Trade extremely large volumes, often entering long positions during periods of fear.

4. **Predictive Machine Learning**:
   * A Random Forest Classifier trained on sentiment values, trade sizing, and timing features predicts trade profitability with **72% accuracy** and a **0.7897 ROC-AUC score**.

---

## 📁 Repository Structure

* **`main.ipynb`**: The primary Jupyter Notebook containing the clean, well-commented student analysis, data visualizations, and results.
* **`run_analysis.py`**: A modular python script that executes the complete analytics pipeline and generates the plot images.
* **`plots/`**: Folder containing all generated visualization charts (saved as PNG assets):
  * `fgi_trend_2024.png` (Fear & Greed trend)
  * `trade_volume_vs_sentiment.png` (Trade count and volume compared to sentiment classification)
  * `activity_heatmap.png` (Heatmap of execution hour vs day of week)
  * `pnl_by_sentiment.png` (Mean PnL across sentiment zones)
  * `directional_ratios.png` (Long vs Short propensity)
  * `trader_clusters.png` (K-Means scatter plot of trader archetypes)
  * `top_vs_bottom_performance.png` (PnL comparisons of top vs bottom traders)
  * `backtest_returns.png` (Cumulative performance of backtest simulations)
  * `feature_importance.png` (Random Forest feature importances)

---

## 🚀 How to Run

### 1. Prerequisites
Install the required packages in your active environment using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Run the Command Line Script
Run the pipeline to output metrics to the console and regenerate the visual assets:
```bash
python run_analysis.py
```

### 3. Open the Jupyter Notebook
Open `main.ipynb` in VS Code or Jupyter, select your virtual environment as the Python kernel, and run all cells to explore the interactive code and inline visualizations.
