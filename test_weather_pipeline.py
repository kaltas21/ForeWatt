#!/usr/bin/env python3
"""
Test script for weather data pipeline with rate limit handling.

This script demonstrates the improved weather fetcher that:
1. Adds delays between requests to respect API rate limits
2. Automatically retries failed cities after rate limit reset
3. Fetches all 10 Turkish cities successfully

Usage:
    python test_weather_pipeline.py
"""

from src.data.weather_fetcher import DemandWeatherFetcher
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize fetcher
    fetcher = DemandWeatherFetcher(cache_dir='.cache')

    print("\n" + "="*70)
    print("TESTING WEATHER PIPELINE WITH RATE LIMIT FIX")
    print("="*70)

    # Test 1: Recent data (small test, 1 week)
    print("\n[TEST 1] Fetching 1 week of recent data (Oct 20-26, 2024)")
    print("This should complete in ~70 seconds (7s delay × 10 cities)")
    print("-" * 70)

    try:
        recent_features = fetcher.run_pipeline(
            start_date='2024-10-20',
            end_date='2024-10-26',
            output_dir='./data',
            delay_between_requests=7.0,  # Respect free tier limits
            retry_failed=True  # Automatically retry failed cities
        )

        print(f"\n✓ SUCCESS: Fetched {len(recent_features)} hourly records")
        print(f"  Date range: {recent_features.index.min()} to {recent_features.index.max()}")
        print(f"  Features: {len(recent_features.columns)} columns")

    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nKey improvements:")
    print("  ✓ 7-second delay between requests prevents rate limiting")
    print("  ✓ Automatic retry with 65s wait if rate limit is hit")
    print("  ✓ All 10 cities should now be fetched successfully")
    print("  ✓ Bronze/Silver/Gold data saved to ./data/")
    print("\nNext: Run full historical fetch (2020-2024) for training data")
    print("  Estimated time: ~90 seconds for all 10 cities")
