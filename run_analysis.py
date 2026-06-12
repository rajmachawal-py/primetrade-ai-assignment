import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def run_pipeline():
    # Set plotting style to match main.ipynb
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Create directory for saving plots
    os.makedirs('plots', exist_ok=True)
    print("--------------------------------------------------")
    print("Primetrade.ai DS Pipeline Initialized")
    print("--------------------------------------------------\n")
    
    # Load data
    print("Loading datasets...")
    trades = pd.read_csv('historical_data.csv')
    fgi = pd.read_csv('fear_greed_index.csv')
    
    # Preprocessing
    trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trades['date'] = trades['Timestamp IST'].dt.date
    fgi['date'] = pd.to_datetime(fgi['date']).dt.date
    
    start_date = pd.to_datetime('2024-01-01').date()
    end_date = pd.to_datetime('2024-12-31').date()
    trades_2024 = trades[(trades['date'] >= start_date) & (trades['date'] <= end_date)].copy()
    
    numeric_fields = ['Closed PnL', 'Size USD', 'Execution Price', 'Fee']
    for col in numeric_fields:
        trades_2024[col] = pd.to_numeric(trades_2024[col], errors='coerce').fillna(0.0)
        
    merged_df = pd.merge(trades_2024, fgi[['date', 'value', 'classification']], on='date', how='inner')
    merged_df['is_profitable'] = merged_df['Closed PnL'] > 0
    
    def get_size_bracket(usd):
        if usd <= 500: return 'Small (<$500)'
        elif usd <= 5000: return 'Medium ($500-$5k)'
        elif usd <= 25000: return 'Large ($5k-$25k)'
        else: return 'Whale (>$25k)'
        
    merged_df['size_bracket'] = merged_df['Size USD'].apply(get_size_bracket)
    merged_df['hour'] = merged_df['Timestamp IST'].dt.hour
    merged_df['day_of_week'] = merged_df['Timestamp IST'].dt.day_name()
    
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    merged_df['classification'] = pd.Categorical(merged_df['classification'], categories=sentiment_order, ordered=True)
    
    # Plot 1: FGI Trend
    fgi_2024 = fgi[(fgi['date'] >= start_date) & (fgi['date'] <= end_date)].sort_values('date')
    plt.figure(figsize=(14, 5))
    plt.plot(fgi_2024['date'], fgi_2024['value'], color='purple', label='FGI')
    plt.axhline(25, color='red', linestyle='--', label='Fear')
    plt.axhline(75, color='green', linestyle='--', label='Greed')
    plt.title('Bitcoin Fear & Greed Index in 2024')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/fgi_trend_2024.png', dpi=300)
    plt.close()
    
    # Plot 2: Trades & Volume
    sentiment_counts = merged_df['classification'].value_counts().reindex(sentiment_order)
    sentiment_volume = merged_df.groupby('classification', observed=False)['Size USD'].sum().reindex(sentiment_order)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Market Sentiment')
    ax1.set_ylabel('Number of Trades', color='blue')
    ax1.bar(sentiment_order, sentiment_counts, color='blue', alpha=0.5, label='Trades')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Volume (USD Millions)', color='red')
    ax2.plot(sentiment_order, sentiment_volume / 1e6, color='red', marker='o', label='Volume')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Trades & Volume across different Sentiments')
    plt.tight_layout()
    plt.savefig('plots/trade_volume_vs_sentiment.png', dpi=300)
    plt.close()
    
    # Plot 3: Heatmap of activity
    hourly_day_activity = merged_df.groupby(['day_of_week', 'hour']).size().unstack()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hourly_day_activity = hourly_day_activity.reindex(days_order)
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(hourly_day_activity, cmap='YlGnBu')
    plt.title('Trading Activity Heatmap: Hour of Day (IST) vs Day of Week')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig('plots/activity_heatmap.png', dpi=300)
    plt.close()
    
    # Stats summary
    sentiment_summary = merged_df.groupby('classification', observed=False).agg(
        total_pnl=('Closed PnL', 'sum'),
        mean_pnl=('Closed PnL', 'mean'),
        median_pnl=('Closed PnL', 'median'),
        trade_count=('Closed PnL', 'count'),
        win_rate=('is_profitable', lambda x: x.mean() * 100)
    ).reindex(sentiment_order)
    
    print("\nPerformance Summary:")
    print(sentiment_summary.to_string())
    
    # Plot 4: Mean PnL & Win Rate
    # In run_analysis.py, we save the Mean PnL to pnl_by_sentiment.png to remain in sync with visual expectations
    plt.figure(figsize=(10, 4))
    sns.barplot(data=sentiment_summary.reset_index(), x='classification', y='mean_pnl', hue='classification', palette='coolwarm', legend=False)
    plt.axhline(0, color='black')
    plt.title('Average PnL per Trade by Sentiment Category')
    plt.tight_layout()
    plt.savefig('plots/pnl_by_sentiment.png', dpi=300)
    plt.close()
    
    # Optional/Faceted comparison plot for top vs bottom traders
    top_5_accounts = merged_df.groupby('Account')['Closed PnL'].sum().nlargest(5).index.tolist()
    bottom_5_accounts = merged_df.groupby('Account')['Closed PnL'].sum().nsmallest(5).index.tolist()
    
    merged_df['trader_cohort'] = 'General'
    merged_df.loc[merged_df['Account'].isin(top_5_accounts), 'trader_cohort'] = 'Top 5 Traders'
    merged_df.loc[merged_df['Account'].isin(bottom_5_accounts), 'trader_cohort'] = 'Bottom 5 Traders'
    
    cohort_sentiment = merged_df[merged_df['trader_cohort'] != 'General'].groupby(
        ['trader_cohort', 'classification'], observed=False
    ).agg(
        avg_pnl=('Closed PnL', 'mean')
    ).reset_index()
    
    g = sns.FacetGrid(cohort_sentiment, col="trader_cohort", height=5, aspect=1.2, sharey=False)
    g.map_dataframe(sns.barplot, x="classification", y="avg_pnl", hue="classification", palette="icefire", legend=False)
    g.set_axis_labels("Market Sentiment", "Average Trade PnL ($)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.axhline(0, color='black', linewidth=1)
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle('Performance Response by Sentiment Regime: Top vs Bottom Traders', y=1.05, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/top_vs_bottom_performance.png', dpi=300)
    plt.close()
    
    # Stats test
    groups = []
    for name, group in merged_df.groupby('classification', observed=False):
        groups.append(group['Closed PnL'].values)
    groups = [g for g in groups if len(g) > 0]
    
    f_stat, anova_p = stats.f_oneway(*groups)
    h_stat, kruskal_p = stats.kruskal(*groups)
    print(f"\nANOVA test p-value: {anova_p:.4e}")
    print(f"Kruskal-Wallis test p-value: {kruskal_p:.4e}")
    
    # Directional bias
    merged_df['is_long'] = merged_df['Direction'].str.lower().str.contains('long|buy', na=False)
    merged_df['is_short'] = merged_df['Direction'].str.lower().str.contains('short|sell', na=False)
    
    sentiment_bias = merged_df.groupby('classification', observed=False).agg(
        total_trades=('Trade ID', 'count'),
        long_trades=('is_long', 'sum'),
        short_trades=('is_short', 'sum')
    ).reindex(sentiment_order)
    
    sentiment_bias['long_percentage'] = (sentiment_bias['long_trades'] / sentiment_bias['total_trades']) * 100
    sentiment_bias['short_percentage'] = (sentiment_bias['short_trades'] / sentiment_bias['total_trades']) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_order, sentiment_bias['long_percentage'], marker='o', color='green', label='Long Percentage')
    plt.plot(sentiment_order, sentiment_bias['short_percentage'], marker='x', color='red', label='Short Percentage')
    plt.title('Propensity of Long/Short Trades across sentiments')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Percentage of Trades (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/directional_ratios.png', dpi=300)
    plt.close()
    
    # Profiles
    trader_profiles = merged_df.groupby('Account').agg(
        trade_count=('Trade ID', 'count'),
        total_pnl=('Closed PnL', 'sum'),
        mean_pnl=('Closed PnL', 'mean'),
        win_rate=('is_profitable', lambda x: x.mean() * 100),
        total_volume=('Size USD', 'sum'),
        long_ratio=('is_long', lambda x: x.mean() * 100)
    ).reset_index()
    
    def get_sentiment_sensitivity(group):
        daily = group.groupby('date').agg({'Size USD': 'sum', 'value': 'first'})
        if len(daily) > 3:
            corr, _ = stats.pearsonr(daily['Size USD'], daily['value'])
            return corr if not np.isnan(corr) else 0.0
        return 0.0
        
    sens_df = merged_df.groupby('Account').apply(get_sentiment_sensitivity, include_groups=False).reset_index()
    sens_df.columns = ['Account', 'sentiment_sensitivity']
    trader_profiles = pd.merge(trader_profiles, sens_df, on='Account')
    
    # KMeans
    features = ['mean_pnl', 'win_rate', 'total_volume', 'long_ratio', 'sentiment_sensitivity']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trader_profiles[features])
    kmeans = KMeans(n_clusters=3, random_state=42)
    trader_profiles['Cluster'] = kmeans.fit_predict(scaled_features)
    
    stats_summary = trader_profiles.groupby('Cluster').agg(
        win_rate=('win_rate', 'mean'),
        volume=('total_volume', 'mean'),
        sensitivity=('sentiment_sensitivity', 'mean')
    ).reset_index()
    
    whale_idx = stats_summary['volume'].idxmax()
    sens_idx = stats_summary['sensitivity'].idxmax()
    
    other_idx = [i for i in [0, 1, 2] if i != whale_idx]
    if len(other_idx) == 2:
        if stats_summary.loc[other_idx[0], 'win_rate'] > stats_summary.loc[other_idx[1], 'win_rate']:
            consistent_idx = other_idx[0]
            momentum_idx = other_idx[1]
        else:
            consistent_idx = other_idx[1]
            momentum_idx = other_idx[0]
    else:
        momentum_idx = sens_idx
        consistent_idx = [i for i in [0, 1, 2] if i != whale_idx and i != momentum_idx][0]
        
    archetype_map = {
        whale_idx: "Contrarian Whale (High Volume & Long-Biased)",
        momentum_idx: "Momentum Follower (High Sentiment Sensitivity)",
        consistent_idx: "Balanced Specialist (Conservative & Consistent)"
    }
    trader_profiles['Archetype'] = trader_profiles['Cluster'].map(archetype_map)
    print("\nTrader Archetypes counts:")
    print(trader_profiles['Archetype'].value_counts().to_string())
    
    # Plot 5: Trader clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=trader_profiles, 
        x='win_rate', 
        y='total_pnl', 
        hue='Archetype', 
        size='total_volume', 
        sizes=(50, 400),
        palette='Set1'
    )
    plt.title('Trader Clustering Analysis')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Total PnL ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/trader_clusters.png', dpi=300)
    plt.close()
    
    # Backtest
    daily_btc = merged_df[merged_df['Coin'] == 'BTC'].copy()
    if len(daily_btc) > 30:
        daily_data = daily_btc.groupby('date')['Execution Price'].mean().reset_index()
    else:
        daily_data = merged_df.groupby('date')['Execution Price'].mean().reset_index()
        
    daily_data['return'] = daily_data['Execution Price'].pct_change().fillna(0.0)
    daily_strat = pd.merge(daily_data, fgi_2024[['date', 'value']], on='date', how='inner')
    
    daily_strat['contrarian_sig'] = 0
    daily_strat.loc[daily_strat['value'] <= 35, 'contrarian_sig'] = 1
    daily_strat.loc[daily_strat['value'] >= 65, 'contrarian_sig'] = -1
    daily_strat['trend_sig'] = -daily_strat['contrarian_sig']
    
    daily_strat['contrarian_ret'] = daily_strat['contrarian_sig'].shift(1) * daily_strat['return']
    daily_strat['trend_ret'] = daily_strat['trend_sig'].shift(1) * daily_strat['return']
    daily_strat['buy_and_hold'] = daily_strat['return']
    
    daily_strat['cum_contrarian'] = (1 + daily_strat['contrarian_ret'].fillna(0.0)).cumprod() - 1
    daily_strat['cum_trend'] = (1 + daily_strat['trend_ret'].fillna(0.0)).cumprod() - 1
    daily_strat['cum_bh'] = (1 + daily_strat['buy_and_hold'].fillna(0.0)).cumprod() - 1
    
    print("\nBacktest performance:")
    print(f"-> Contrarian Strategy Return: {daily_strat['cum_contrarian'].iloc[-1]*100:.2f}%")
    print(f"-> Trend Following Strategy Return: {daily_strat['cum_trend'].iloc[-1]*100:.2f}%")
    print(f"-> Buy & Hold Return: {daily_strat['cum_bh'].iloc[-1]*100:.2f}%")
    
    # Plot 6: Backtest
    plt.figure(figsize=(12, 6))
    plt.plot(daily_strat['date'], daily_strat['cum_contrarian'] * 100, label='Contrarian strategy')
    plt.plot(daily_strat['date'], daily_strat['cum_trend'] * 100, label='Trend Following')
    plt.plot(daily_strat['date'], daily_strat['cum_bh'] * 100, label='Buy & Hold Benchmark', linestyle='--')
    plt.title('Backtesting Sentiment Trading Strategies (2024)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/backtest_returns.png', dpi=300)
    plt.close()
    
    # Random Forest Model
    print("\n[6/6] Fitting predictive machine learning model...")
    ml_cols = ['value', 'Size USD', 'hour', 'is_profitable', 'Side']
    df_ml = merged_df[ml_cols].dropna().copy()
    df_ml = pd.get_dummies(df_ml, columns=['Side'], drop_first=True)
    
    X = df_ml.drop('is_profitable', axis=1)
    y = df_ml['is_profitable']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_preds = rf_model.predict(X_test)
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    
    print("\nMachine Learning Classification Report:")
    print(classification_report(y_test, y_preds))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
    
    # Save Plot 7: Feature Importances
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='darkcyan')
    plt.title('Feature Importances for Profitability Classification')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    plt.close()
    
    print("\n--------------------------------------------------")
    print("Pipeline Execution Completed. All plots saved to plots/")
    print("--------------------------------------------------")

if __name__ == '__main__':
    run_pipeline()
