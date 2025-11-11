"""
Deflation Pipeline Validation Script
=====================================
Tests the complete TL normalization pipeline with both synthetic and real data.

Validation Steps:
1. Check prerequisites (API keys, dependencies)
2. Create synthetic test data (if real data unavailable)
3. Test EVDS fetcher
4. Test deflator builder (baseline + DFM)
5. Test interpolation (monthly → daily → hourly)
6. Test price deflation
7. Validate output quality (stationarity, variance reduction, etc.)

Usage:
    # Dry run (synthetic data only)
    python src/data/validate_deflation_pipeline.py --dry-run

    # Full pipeline (requires API keys)
    python src/data/validate_deflation_pipeline.py --full

    # Test specific component
    python src/data/validate_deflation_pipeline.py --test interpolation
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeflationPipelineValidator:
    """Validates the complete deflation pipeline."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.test_results = {}

    def run_all_tests(self):
        """Run complete validation suite."""
        logger.info("=" * 70)
        logger.info("DEFLATION PIPELINE VALIDATION")
        logger.info("=" * 70)

        tests = [
            ("Prerequisites", self.test_prerequisites),
            ("Synthetic Data", self.test_synthetic_data),
            ("EVDS Fetcher", self.test_evds_fetcher),
            ("Deflator Builder", self.test_deflator_builder),
            ("Interpolation", self.test_interpolation),
            ("Price Deflation", self.test_price_deflation),
            ("Output Quality", self.test_output_quality),
        ]

        for test_name, test_func in tests:
            logger.info(f"\n{'─' * 70}")
            logger.info(f"TEST: {test_name}")
            logger.info(f"{'─' * 70}")

            try:
                result = test_func()
                self.test_results[test_name] = result

                if result['status'] == 'pass':
                    logger.info(f"✅ PASSED: {result.get('message', 'OK')}")
                elif result['status'] == 'skip':
                    logger.info(f"⏭️  SKIPPED: {result.get('message', 'N/A')}")
                else:
                    logger.warning(f"⚠️  WARNING: {result.get('message', 'Issues detected')}")

            except Exception as e:
                logger.error(f"❌ FAILED: {str(e)}")
                self.test_results[test_name] = {'status': 'fail', 'error': str(e)}

        # Print summary
        self._print_summary()

    def test_prerequisites(self):
        """Check that required dependencies and configs are present."""
        issues = []

        # Check Python packages
        required_packages = [
            'pandas', 'numpy', 'evds', 'statsmodels',
            'sklearn', 'pyarrow', 'python-dotenv'
        ]

        # Map import names to package names (some differ)
        package_import_map = {
            'python-dotenv': 'dotenv'
        }

        for package in required_packages:
            try:
                # Use mapped import name if available
                import_name = package_import_map.get(package, package)
                __import__(import_name)
            except ImportError:
                issues.append(f"Missing package: {package}")

        # Check .env file
        if not Path('.env').exists():
            issues.append("Missing .env file (not critical for dry-run)")
        else:
            # Check for EVDS_API_KEY
            with open('.env', 'r') as f:
                env_content = f.read()
                if 'EVDS_API_KEY' not in env_content:
                    issues.append("EVDS_API_KEY not in .env (not critical for dry-run)")
                elif 'your_evds_api_key_here' in env_content:
                    issues.append("EVDS_API_KEY not configured in .env (not critical for dry-run)")

        # Check directory structure (create if missing)
        required_dirs = [
            'data/bronze/epias',
            'data/silver/epias',
            'data/gold/epias'
        ]

        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Separate critical vs non-critical issues
        critical_issues = [i for i in issues if 'not critical' not in i]
        non_critical_issues = [i for i in issues if 'not critical' in i]

        if critical_issues:
            logger.warning(f"  Critical issues found:")
            for issue in critical_issues:
                logger.warning(f"    - {issue}")
            return {
                'status': 'warning',
                'message': f"Found {len(critical_issues)} critical issue(s)",
                'issues': critical_issues
            }

        if non_critical_issues:
            logger.info(f"  Note: {len(non_critical_issues)} non-critical issue(s) (OK for dry-run):")
            for issue in non_critical_issues:
                logger.info(f"    - {issue}")

        return {'status': 'pass', 'message': 'All prerequisites met'}

    def test_synthetic_data(self):
        """Create synthetic test data for pipeline validation."""
        logger.info("Creating synthetic macro data...")

        # Generate monthly data for 2020-2024 (full coverage)
        # Start from 2020-01 to match EPİAŞ data range
        date_range = pd.date_range('2020-01-01', '2024-12-31', freq='MS')

        # Synthetic EVDS data (realistic Turkish inflation patterns)
        np.random.seed(42)

        # Base inflation trend (moderate 2020 → high inflation 2021-2024)
        # 2020: ~12% annual, 2021-2022: ~20-40%, 2023-2024: ~60-70%
        base_inflation = np.array([12] * 12 +  # 2020
                                   [20] * 12 +  # 2021
                                   [35] * 12 +  # 2022
                                   [60] * 12 +  # 2023
                                   [70] * 12)   # 2024
        noise = np.random.normal(0, 3, len(date_range))

        tufe = 100 * np.exp(np.cumsum((base_inflation + noise) / 1200))  # CPI
        ufe = tufe * np.random.uniform(0.95, 1.05, len(date_range))  # PPI
        m2 = np.random.uniform(15, 30, len(date_range))  # M2 growth
        tl_faiz = base_inflation + np.random.uniform(-5, 5, len(date_range))  # Interest rate

        synthetic_macro = pd.DataFrame({
            'DATE': date_range.strftime('%Y-%m'),
            'TUFE': tufe,
            'UFE': ufe,
            'M2': m2,
            'TL_FAIZ': tl_faiz
        })

        # Save to bronze layer
        bronze_dir = Path('data/bronze/macro')
        bronze_dir.mkdir(parents=True, exist_ok=True)

        output_path = bronze_dir / 'macro_evds_2020-01-01_2024-12-31_SYNTHETIC.parquet'
        synthetic_macro.to_parquet(output_path, engine='pyarrow', compression='snappy')

        logger.info(f"  Created {len(synthetic_macro)} months of synthetic data")
        logger.info(f"  Saved to: {output_path}")

        return {
            'status': 'pass',
            'message': f'Created {len(synthetic_macro)} synthetic data points',
            'data': synthetic_macro
        }

    def test_evds_fetcher(self):
        """Test EVDS data fetching."""
        if self.dry_run:
            logger.info("  Using synthetic data (dry-run mode)")
            logger.info("  To test with real data: python validate_deflation_pipeline.py --full")
            return {'status': 'skip', 'message': 'Dry run mode, using synthetic data'}

        # Check if EVDS_API_KEY is configured
        from dotenv import load_dotenv
        load_dotenv()

        if not os.getenv('EVDS_API_KEY') or os.getenv('EVDS_API_KEY') == 'your_evds_api_key_here':
            logger.warning("  EVDS_API_KEY not configured")
            logger.info("  Get API key from: https://evds2.tcmb.gov.tr/")
            logger.info("  Add to .env: EVDS_API_KEY=your_key_here")
            return {
                'status': 'skip',
                'message': 'EVDS_API_KEY not configured. Get key from https://evds2.tcmb.gov.tr/'
            }

        try:
            # Import and run evds_fetcher
            sys.path.insert(0, str(Path('src/data')))
            from evds_fetcher import fetch_evds_data, save_bronze

            logger.info("Fetching real EVDS data (2020-2024)...")
            df = fetch_evds_data(start_date='2020-01-01', end_date='2024-12-31')

            # Validate data
            required_cols = ['DATE', 'TUFE', 'UFE']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                return {
                    'status': 'fail',
                    'message': f'Missing required columns: {missing_cols}'
                }

            # Save bronze
            save_bronze(df, start_date='2020-01-01', end_date='2024-12-31')

            return {
                'status': 'pass',
                'message': f'Fetched {len(df)} months of real EVDS data',
                'data': df
            }

        except Exception as e:
            return {
                'status': 'fail',
                'message': f'EVDS fetch failed: {str(e)}'
            }

    def test_deflator_builder(self):
        """Test deflator index construction."""
        logger.info("Building deflator indices...")

        # Check if macro data exists (real or synthetic)
        bronze_dir = Path('data/bronze/macro')

        # Try to find any macro data file
        macro_files = list(bronze_dir.glob('macro_evds_*.parquet'))
        if not macro_files:
            macro_files = list(bronze_dir.glob('macro_evds_*.csv'))

        if not macro_files:
            return {
                'status': 'fail',
                'message': 'No macro data found. Run EVDS fetcher first.'
            }

        try:
            sys.path.insert(0, str(Path('src/data')))
            from deflator_builder import build_did_baseline, build_did_dfm

            # Test baseline deflator
            logger.info("  Building baseline DID (Factor Analysis)...")
            build_did_baseline()

            # Check output
            baseline_file = Path('data/silver/macro/deflator_did_baseline.parquet')
            if not baseline_file.exists():
                baseline_file = Path('data/silver/macro/deflator_did_baseline.csv')

            if not baseline_file.exists():
                return {
                    'status': 'fail',
                    'message': 'Baseline deflator file not created'
                }

            # Load and validate
            if baseline_file.suffix == '.parquet':
                deflator_df = pd.read_parquet(baseline_file)
            else:
                deflator_df = pd.read_csv(baseline_file)

            # Validate structure
            if 'DID_index' not in deflator_df.columns:
                return {
                    'status': 'fail',
                    'message': 'DID_index column missing from deflator'
                }

            # Test DFM deflator (optional, more complex)
            try:
                logger.info("  Building DFM DID (Kalman smoothing)...")
                build_did_dfm()
            except Exception as e:
                logger.warning(f"  DFM deflator failed (not critical): {e}")

            return {
                'status': 'pass',
                'message': f'Built deflator with {len(deflator_df)} data points',
                'data': deflator_df
            }

        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Deflator builder failed: {str(e)}'
            }

    def test_interpolation(self):
        """Test monthly → daily → hourly interpolation."""
        logger.info("Testing interpolation logic...")

        # Load deflator
        deflator_file = Path('data/silver/macro/deflator_did_baseline.parquet')
        if not deflator_file.exists():
            deflator_file = Path('data/silver/macro/deflator_did_baseline.csv')

        if not deflator_file.exists():
            return {
                'status': 'skip',
                'message': 'No deflator file found. Run deflator builder first.'
            }

        try:
            sys.path.insert(0, str(Path('src/data')))
            from deflate_prices import PriceDeflator

            # Initialize deflator (triggers interpolation)
            deflator = PriceDeflator(deflator_method='baseline')

            hourly_df = deflator.deflator_df

            # Validate interpolation
            if hourly_df is None or len(hourly_df) == 0:
                return {
                    'status': 'fail',
                    'message': 'Interpolation produced empty result'
                }

            # Check for gaps
            time_diffs = hourly_df['datetime'].diff()
            gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]

            if len(gaps) > 0:
                logger.warning(f"  Found {len(gaps)} time gaps in interpolated data")

            # Check for smoothness (daily changes should be small)
            daily_values = hourly_df.set_index('datetime')['DID_index'].resample('D').first()
            daily_changes = daily_values.pct_change().abs()
            max_daily_change = daily_changes.max()

            logger.info(f"  Interpolated to {len(hourly_df)} hourly values")
            logger.info(f"  Max daily DID change: {max_daily_change:.2%}")

            if max_daily_change > 0.05:
                logger.warning(f"  Large daily change detected: {max_daily_change:.2%}")

            return {
                'status': 'pass',
                'message': f'Interpolated {len(hourly_df)} hourly values',
                'data': hourly_df,
                'max_daily_change': max_daily_change
            }

        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Interpolation failed: {str(e)}'
            }

    def test_price_deflation(self):
        """Test price deflation on real EPİAŞ data."""
        logger.info("Testing price deflation...")

        # Check if silver price data exists
        price_file = Path('data/silver/epias/price_ptf_normalized_2020-01-01_2024-12-31.parquet')
        if not price_file.exists():
            return {
                'status': 'skip',
                'message': 'No silver price data found. Run epias_fetcher.py first.'
            }

        try:
            sys.path.insert(0, str(Path('src/data')))
            from deflate_prices import PriceDeflator

            # Initialize and deflate PTF
            deflator = PriceDeflator(deflator_method='baseline')

            logger.info("  Deflating price_ptf dataset...")
            df_deflated = deflator.deflate_dataset(
                dataset_name='price_ptf',
                start_date='2020-01-01',
                end_date='2024-12-31',
                layer='silver'
            )

            # Validate output
            real_cols = [col for col in df_deflated.columns if '_real' in col]

            if not real_cols:
                return {
                    'status': 'fail',
                    'message': 'No *_real columns created during deflation'
                }

            logger.info(f"  Created {len(real_cols)} real-value column(s): {real_cols}")

            # Check variance reduction
            nominal_col = real_cols[0].replace('_real', '')
            if nominal_col in df_deflated.columns:
                var_nominal = df_deflated[nominal_col].var()
                var_real = df_deflated[real_cols[0]].var()
                variance_reduction = (1 - var_real / var_nominal) * 100

                logger.info(f"  Variance reduction: {variance_reduction:.1f}%")

                if variance_reduction < 10:
                    logger.warning("  Low variance reduction (expected 20-40%)")

            return {
                'status': 'pass',
                'message': f'Deflated {len(df_deflated)} records',
                'data': df_deflated,
                'variance_reduction': variance_reduction if 'variance_reduction' in locals() else None
            }

        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Price deflation failed: {str(e)}'
            }

    def test_output_quality(self):
        """Validate statistical properties of deflated prices."""
        logger.info("Validating output quality...")

        # Check if gold deflated data exists
        deflated_file = Path('data/gold/epias/price_ptf_deflated_2020-01-01_2024-12-31.parquet')

        if not deflated_file.exists():
            return {
                'status': 'skip',
                'message': 'No deflated data found. Run deflation first.'
            }

        try:
            df = pd.read_parquet(deflated_file)

            # Find real price column
            real_cols = [col for col in df.columns if '_real' in col]
            if not real_cols:
                return {'status': 'fail', 'message': 'No *_real columns found'}

            price_real = df[real_cols[0]].dropna()

            # Test 1: Check for negative values
            negative_count = (price_real < 0).sum()
            if negative_count > 0:
                logger.warning(f"  Found {negative_count} negative real prices")

            # Test 2: Check for extreme outliers (> 5 std from mean)
            mean_price = price_real.mean()
            std_price = price_real.std()
            outliers = price_real[(price_real > mean_price + 5 * std_price) |
                                 (price_real < mean_price - 5 * std_price)]

            if len(outliers) > len(price_real) * 0.01:  # > 1% outliers
                logger.warning(f"  Found {len(outliers)} extreme outliers")

            # Test 3: Stationarity check (Augmented Dickey-Fuller)
            try:
                from statsmodels.tsa.stattools import adfuller

                # Test with reasonable sample (avoid memory issues)
                sample_size = min(10000, len(price_real))
                adf_result = adfuller(price_real.sample(sample_size, random_state=42))
                p_value = adf_result[1]

                logger.info(f"  ADF test p-value: {p_value:.4f}")

                if p_value < 0.05:
                    logger.info("  ✓ Real prices are stationary (p < 0.05)")
                else:
                    logger.warning("  Real prices not stationary (p >= 0.05)")

            except Exception as e:
                logger.warning(f"  Could not run stationarity test: {e}")

            # Summary statistics
            logger.info(f"  Mean real price: {mean_price:.2f} TL/MWh")
            logger.info(f"  Std real price: {std_price:.2f} TL/MWh")
            logger.info(f"  Min/Max: {price_real.min():.2f} / {price_real.max():.2f}")

            return {
                'status': 'pass',
                'message': 'Output quality validated',
                'stats': {
                    'mean': mean_price,
                    'std': std_price,
                    'negative_count': negative_count,
                    'outlier_count': len(outliers)
                }
            }

        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Quality validation failed: {str(e)}'
            }

    def _print_summary(self):
        """Print validation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)

        passed = sum(1 for r in self.test_results.values() if r['status'] == 'pass')
        skipped = sum(1 for r in self.test_results.values() if r['status'] == 'skip')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'fail')
        warnings = sum(1 for r in self.test_results.values() if r['status'] == 'warning')

        total = len(self.test_results)

        logger.info(f"Total tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"⏭️  Skipped: {skipped}")
        logger.info(f"⚠️  Warnings: {warnings}")
        logger.info(f"❌ Failed: {failed}")

        if failed == 0 and warnings == 0:
            logger.info("\n🎉 All tests passed successfully!")
        elif failed == 0:
            logger.info(f"\n⚠️  Pipeline functional with {warnings} warning(s)")
        else:
            logger.info(f"\n❌ Pipeline validation failed ({failed} test(s))")

        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Validate deflation pipeline')
    parser.add_argument('--dry-run', action='store_true',
                       help='Use synthetic data only (no API calls)')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline with real data')
    parser.add_argument('--test', type=str, choices=[
        'prerequisites', 'synthetic', 'evds', 'deflator',
        'interpolation', 'deflation', 'quality'
    ], help='Run specific test only')

    args = parser.parse_args()

    validator = DeflationPipelineValidator(dry_run=args.dry_run or not args.full)

    if args.test:
        # Run specific test
        test_map = {
            'prerequisites': validator.test_prerequisites,
            'synthetic': validator.test_synthetic_data,
            'evds': validator.test_evds_fetcher,
            'deflator': validator.test_deflator_builder,
            'interpolation': validator.test_interpolation,
            'deflation': validator.test_price_deflation,
            'quality': validator.test_output_quality,
        }

        logger.info(f"Running test: {args.test}")
        result = test_map[args.test]()
        logger.info(f"Result: {result}")
    else:
        # Run all tests
        validator.run_all_tests()


if __name__ == '__main__':
    main()
