#!/usr/bin/env python3
"""
Zone Fade Entry Points Visualization Script

This script creates comprehensive visualizations for the Zone Fade entry points data
from the 2024 efficient backtesting results.

Features:
- Time series analysis
- QRS score distributions
- Symbol-based analysis
- Zone type and quality analysis
- Entry window duration analysis
- Price deviation analysis
- Volume spike analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ZoneFadeVisualizer:
    """Visualization class for Zone Fade entry points data."""
    
    def __init__(self, csv_path):
        """Initialize with CSV data."""
        self.csv_path = csv_path
        self.df = self.load_data()
        self.setup_plotting()
        
    def load_data(self):
        """Load and preprocess the CSV data."""
        print("Loading Zone Fade entry points data...")
        df = pd.read_csv(self.csv_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Convert boolean columns
        df['rejection_candle'] = df['rejection_candle'].astype(bool)
        df['volume_spike'] = df['volume_spike'].astype(bool)
        df['entry_window_ended'] = df['entry_window_ended'].astype(bool)
        
        print(f"Loaded {len(df)} entry points")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Symbols: {df['symbol'].unique()}")
        
        return df
    
    def setup_plotting(self):
        """Set up plotting parameters."""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_overview_dashboard(self):
        """Create an overview dashboard with key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Zone Fade Entry Points - Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Entry points by symbol
        symbol_counts = self.df['symbol'].value_counts()
        axes[0, 0].pie(symbol_counts.values, labels=symbol_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Entry Points by Symbol')
        
        # 2. QRS Score distribution
        axes[0, 1].hist(self.df['qrs_score'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.df['qrs_score'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["qrs_score"].mean():.1f}')
        axes[0, 1].set_title('QRS Score Distribution')
        axes[0, 1].set_xlabel('QRS Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Zone types
        zone_counts = self.df['zone_type'].value_counts()
        axes[0, 2].bar(zone_counts.index, zone_counts.values)
        axes[0, 2].set_title('Entry Points by Zone Type')
        axes[0, 2].set_xlabel('Zone Type')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Volume spike analysis
        volume_spike_counts = self.df['volume_spike'].value_counts()
        axes[1, 0].pie(volume_spike_counts.values, labels=['No Volume Spike', 'Volume Spike'], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Volume Spike Analysis')
        
        # 5. Entry window duration
        axes[1, 1].hist(self.df['window_duration_minutes'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(self.df['window_duration_minutes'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["window_duration_minutes"].mean():.1f} min')
        axes[1, 1].set_title('Entry Window Duration')
        axes[1, 1].set_xlabel('Duration (minutes)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # 6. Rejection candle analysis
        rejection_counts = self.df['rejection_candle'].value_counts()
        axes[1, 2].pie(rejection_counts.values, labels=['No Rejection', 'Rejection Candle'], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Rejection Candle Analysis')
        
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_overview_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_time_series_analysis(self):
        """Create time series analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
        
        # 1. Entry points over time by symbol
        for symbol in self.df['symbol'].unique():
            symbol_data = self.df[self.df['symbol'] == symbol]
            axes[0, 0].scatter(symbol_data['timestamp'], symbol_data['qrs_score'], 
                             label=symbol, alpha=0.7, s=50)
        axes[0, 0].set_title('QRS Scores Over Time by Symbol')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('QRS Score')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Monthly distribution
        monthly_counts = self.df.groupby(self.df['timestamp'].dt.to_period('M')).size()
        monthly_counts.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Entry Points by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Hourly distribution
        hourly_counts = self.df['hour'].value_counts().sort_index()
        axes[1, 0].bar(hourly_counts.index, hourly_counts.values)
        axes[1, 0].set_title('Entry Points by Hour of Day')
        axes[1, 0].set_xlabel('Hour (UTC)')
        axes[1, 0].set_ylabel('Count')
        
        # 4. Day of week distribution
        dow_counts = self.df['day_of_week'].value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = dow_counts.reindex(dow_order)
        axes[1, 1].bar(dow_counts.index, dow_counts.values)
        axes[1, 1].set_title('Entry Points by Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_qrs_analysis(self):
        """Create QRS score analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('QRS Score Analysis', fontsize=16, fontweight='bold')
        
        # 1. QRS distribution by symbol
        for symbol in self.df['symbol'].unique():
            symbol_data = self.df[self.df['symbol'] == symbol]['qrs_score']
            axes[0, 0].hist(symbol_data, alpha=0.6, label=symbol, bins=15)
        axes[0, 0].set_title('QRS Score Distribution by Symbol')
        axes[0, 0].set_xlabel('QRS Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. QRS vs Zone Quality
        qrs_zone_quality = self.df.groupby('zone_quality')['qrs_score'].agg(['mean', 'std', 'count'])
        axes[0, 1].bar(qrs_zone_quality.index, qrs_zone_quality['mean'], 
                      yerr=qrs_zone_quality['std'], capsize=5)
        axes[0, 1].set_title('Average QRS Score by Zone Quality')
        axes[0, 1].set_xlabel('Zone Quality')
        axes[0, 1].set_ylabel('Average QRS Score')
        
        # 3. QRS vs Volume Spike
        qrs_volume = self.df.groupby('volume_spike')['qrs_score'].agg(['mean', 'std'])
        axes[1, 0].bar(['No Volume Spike', 'Volume Spike'], qrs_volume['mean'], 
                      yerr=qrs_volume['std'], capsize=5)
        axes[1, 0].set_title('Average QRS Score by Volume Spike')
        axes[1, 0].set_ylabel('Average QRS Score')
        
        # 4. QRS vs Rejection Candle
        qrs_rejection = self.df.groupby('rejection_candle')['qrs_score'].agg(['mean', 'std'])
        axes[1, 1].bar(['No Rejection', 'Rejection Candle'], qrs_rejection['mean'], 
                      yerr=qrs_rejection['std'], capsize=5)
        axes[1, 1].set_title('Average QRS Score by Rejection Candle')
        axes[1, 1].set_ylabel('Average QRS Score')
        
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_qrs_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_symbol_analysis(self):
        """Create symbol-specific analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Symbol Analysis', fontsize=16, fontweight='bold')
        
        # 1. Entry points count by symbol
        symbol_counts = self.df['symbol'].value_counts()
        axes[0, 0].bar(symbol_counts.index, symbol_counts.values)
        axes[0, 0].set_title('Total Entry Points by Symbol')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Average QRS score by symbol
        qrs_by_symbol = self.df.groupby('symbol')['qrs_score'].agg(['mean', 'std'])
        axes[0, 1].bar(qrs_by_symbol.index, qrs_by_symbol['mean'], 
                      yerr=qrs_by_symbol['std'], capsize=5)
        axes[0, 1].set_title('Average QRS Score by Symbol')
        axes[0, 1].set_ylabel('Average QRS Score')
        
        # 3. Zone types by symbol
        zone_symbol = pd.crosstab(self.df['symbol'], self.df['zone_type'])
        zone_symbol.plot(kind='bar', ax=axes[1, 0], stacked=True)
        axes[1, 0].set_title('Zone Types by Symbol')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Zone Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Volume spike rate by symbol
        volume_by_symbol = self.df.groupby('symbol')['volume_spike'].mean() * 100
        axes[1, 1].bar(volume_by_symbol.index, volume_by_symbol.values)
        axes[1, 1].set_title('Volume Spike Rate by Symbol (%)')
        axes[1, 1].set_ylabel('Volume Spike Rate (%)')
        
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_symbol_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_zone_analysis(self):
        """Create zone type and quality analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Zone Analysis', fontsize=16, fontweight='bold')
        
        # 1. Zone types distribution
        zone_counts = self.df['zone_type'].value_counts()
        axes[0, 0].pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Distribution of Zone Types')
        
        # 2. Zone quality distribution
        quality_counts = self.df['zone_quality'].value_counts().sort_index()
        axes[0, 1].bar(quality_counts.index, quality_counts.values)
        axes[0, 1].set_title('Zone Quality Distribution')
        axes[0, 1].set_xlabel('Zone Quality')
        axes[0, 1].set_ylabel('Count')
        
        # 3. QRS score by zone type
        qrs_by_zone = self.df.groupby('zone_type')['qrs_score'].agg(['mean', 'std'])
        axes[1, 0].bar(qrs_by_zone.index, qrs_by_zone['mean'], 
                      yerr=qrs_by_zone['std'], capsize=5)
        axes[1, 0].set_title('Average QRS Score by Zone Type')
        axes[1, 0].set_ylabel('Average QRS Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Zone strength distribution
        axes[1, 1].hist(self.df['zone_strength'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(self.df['zone_strength'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["zone_strength"].mean():.2f}')
        axes[1, 1].set_title('Zone Strength Distribution')
        axes[1, 1].set_xlabel('Zone Strength')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_zone_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_entry_window_analysis(self):
        """Create entry window duration analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Entry Window Analysis', fontsize=16, fontweight='bold')
        
        # 1. Window duration distribution
        axes[0, 0].hist(self.df['window_duration_minutes'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.df['window_duration_minutes'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["window_duration_minutes"].mean():.1f} min')
        axes[0, 0].set_title('Entry Window Duration Distribution')
        axes[0, 0].set_xlabel('Duration (minutes)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Window duration by symbol
        for symbol in self.df['symbol'].unique():
            symbol_data = self.df[self.df['symbol'] == symbol]['window_duration_minutes']
            axes[0, 1].hist(symbol_data, alpha=0.6, label=symbol, bins=15)
        axes[0, 1].set_title('Window Duration by Symbol')
        axes[0, 1].set_xlabel('Duration (minutes)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Window duration vs QRS score
        axes[1, 0].scatter(self.df['window_duration_minutes'], self.df['qrs_score'], alpha=0.6)
        axes[1, 0].set_title('Window Duration vs QRS Score')
        axes[1, 0].set_xlabel('Duration (minutes)')
        axes[1, 0].set_ylabel('QRS Score')
        
        # 4. Window bars distribution
        axes[1, 1].hist(self.df['window_bars'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(self.df['window_bars'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["window_bars"].mean():.1f} bars')
        axes[1, 1].set_title('Window Bars Distribution')
        axes[1, 1].set_xlabel('Number of Bars')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_entry_window_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_price_deviation_analysis(self):
        """Create price deviation analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Price Deviation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Max price deviation distribution
        axes[0, 0].hist(self.df['max_price_deviation'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.df['max_price_deviation'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["max_price_deviation"].mean():.3f}')
        axes[0, 0].set_title('Max Price Deviation Distribution')
        axes[0, 0].set_xlabel('Max Price Deviation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Min price deviation distribution
        axes[0, 1].hist(self.df['min_price_deviation'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.df['min_price_deviation'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["min_price_deviation"].mean():.3f}')
        axes[0, 1].set_title('Min Price Deviation Distribution')
        axes[0, 1].set_xlabel('Min Price Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Price deviation vs QRS score
        axes[1, 0].scatter(self.df['max_price_deviation'], self.df['qrs_score'], alpha=0.6, label='Max Deviation')
        axes[1, 0].scatter(self.df['min_price_deviation'], self.df['qrs_score'], alpha=0.6, label='Min Deviation')
        axes[1, 0].set_title('Price Deviation vs QRS Score')
        axes[1, 0].set_xlabel('Price Deviation')
        axes[1, 0].set_ylabel('QRS Score')
        axes[1, 0].legend()
        
        # 4. Deviation range by symbol
        deviation_by_symbol = self.df.groupby('symbol').agg({
            'max_price_deviation': 'mean',
            'min_price_deviation': 'mean'
        })
        x = np.arange(len(deviation_by_symbol.index))
        width = 0.35
        axes[1, 1].bar(x - width/2, deviation_by_symbol['max_price_deviation'], width, label='Max Deviation')
        axes[1, 1].bar(x + width/2, deviation_by_symbol['min_price_deviation'], width, label='Min Deviation')
        axes[1, 1].set_title('Average Price Deviation by Symbol')
        axes[1, 1].set_xlabel('Symbol')
        axes[1, 1].set_ylabel('Average Deviation')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(deviation_by_symbol.index)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_price_deviation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap of numerical variables."""
        plt.figure(figsize=(12, 10))
        
        # Select numerical columns for correlation
        numerical_cols = ['qrs_score', 'zone_strength', 'zone_quality', 'window_duration_minutes', 
                         'window_bars', 'max_price_deviation', 'min_price_deviation', 'bar_index']
        
        # Create correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/zone_fade_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_statistics(self):
        """Generate and display summary statistics."""
        print("\n" + "="*60)
        print("ZONE FADE ENTRY POINTS - SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal Entry Points: {len(self.df)}")
        print(f"Date Range: {self.df['timestamp'].min().strftime('%Y-%m-%d')} to {self.df['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"Symbols: {', '.join(self.df['symbol'].unique())}")
        
        print(f"\nQRS Score Statistics:")
        print(f"  Mean: {self.df['qrs_score'].mean():.2f}")
        print(f"  Median: {self.df['qrs_score'].median():.2f}")
        print(f"  Std Dev: {self.df['qrs_score'].std():.2f}")
        print(f"  Min: {self.df['qrs_score'].min():.2f}")
        print(f"  Max: {self.df['qrs_score'].max():.2f}")
        
        print(f"\nEntry Window Duration Statistics:")
        print(f"  Mean: {self.df['window_duration_minutes'].mean():.1f} minutes")
        print(f"  Median: {self.df['window_duration_minutes'].median():.1f} minutes")
        print(f"  Std Dev: {self.df['window_duration_minutes'].std():.1f} minutes")
        
        print(f"\nVolume Spike Analysis:")
        volume_spike_rate = self.df['volume_spike'].mean() * 100
        print(f"  Volume Spike Rate: {volume_spike_rate:.1f}%")
        
        print(f"\nRejection Candle Analysis:")
        rejection_rate = self.df['rejection_candle'].mean() * 100
        print(f"  Rejection Candle Rate: {rejection_rate:.1f}%")
        
        print(f"\nZone Type Distribution:")
        for zone_type, count in self.df['zone_type'].value_counts().items():
            percentage = (count / len(self.df)) * 100
            print(f"  {zone_type}: {count} ({percentage:.1f}%)")
        
        print(f"\nSymbol Distribution:")
        for symbol, count in self.df['symbol'].value_counts().items():
            percentage = (count / len(self.df)) * 100
            print(f"  {symbol}: {count} ({percentage:.1f}%)")
    
    def run_all_visualizations(self):
        """Run all visualization methods."""
        print("Creating Zone Fade Entry Points Visualizations...")
        print("="*50)
        
        # Create outputs directory if it doesn't exist
        import os
        os.makedirs('outputs', exist_ok=True)
        
        # Generate summary statistics
        self.generate_summary_statistics()
        
        # Create all visualizations
        print("\nGenerating visualizations...")
        self.create_overview_dashboard()
        self.create_time_series_analysis()
        self.create_qrs_analysis()
        self.create_symbol_analysis()
        self.create_zone_analysis()
        self.create_entry_window_analysis()
        self.create_price_deviation_analysis()
        self.create_correlation_heatmap()
        
        print(f"\nAll visualizations saved to 'outputs/' directory")
        print("Visualization complete!")


def main():
    """Main function to run the visualization."""
    csv_path = 'results/2024/efficient/zone_fade_entry_points_2024_efficient.csv'
    
    # Create visualizer instance
    visualizer = ZoneFadeVisualizer(csv_path)
    
    # Run all visualizations
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()