"""
Main CLI entry point for VAHAN web scraper.
Provides command-line interface for all scraping and processing operations.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.core.exceptions import VahanScraperError
from src.scrapers import VahanScraper
from src.processors import VahanDataProcessor
from src.analytics import GrowthAnalyzer, InsightGenerator
from src.utils import setup_logging, get_logger, FileManager, create_sample_data

def setup_cli_parser():
    """Setup full-featured command line argument parser for VAHAN dashboard."""
    parser = argparse.ArgumentParser(
        description="🚗 VAHAN Vehicle Registration Data Scraper & Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scrape --states Karnataka Maharashtra --years 2023 2024
  python main.py process --input data/raw_data.csv --output processed_data.csv
  python main.py analyze --input processed_data.csv --generate-insights
  python main.py sample --output sample_data.csv --rows 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ----------------------------
    # Scrape Command
    # ----------------------------
    scrape_parser = subparsers.add_parser('scrape', help='Scrape VAHAN data online')
    scrape_parser.add_argument('--states', nargs='+', required=True,
                              help='States to scrape (e.g., Karnataka Maharashtra)')
    scrape_parser.add_argument('--years', nargs='+', required=True,
                              help='Years to scrape (e.g., 2023 2024)')
    scrape_parser.add_argument('--vehicle-types', nargs='*', default=None,
                              help='Optional vehicle types to scrape')
    scrape_parser.add_argument('--output', '-o', default=None,
                              help='Output CSV filename (auto-generated if omitted)')
    scrape_parser.add_argument('--headless', action='store_true', default=True,
                              help='Run browser in headless mode')
    
    # ----------------------------
    # Process Command
    # ----------------------------
    process_parser = subparsers.add_parser('process', help='Process and clean scraped data')
    process_parser.add_argument('--input', '-i', required=True,
                                help='Input CSV file path')
    process_parser.add_argument('--output', '-o', default=None,
                                help='Output CSV filename (auto-generated if omitted)')
    process_parser.add_argument('--export-metrics', action='store_true',
                                help='Export growth metrics separately')
    
    # ----------------------------
    # Analyze Command
    # ----------------------------
    analyze_parser = subparsers.add_parser('analyze', help='Analyze processed data')
    analyze_parser.add_argument('--input', '-i', required=True,
                                help='Input processed CSV file path')
    analyze_parser.add_argument('--output-dir', '-o', default=None,
                                help='Output directory for analysis results')
    analyze_parser.add_argument('--generate-insights', action='store_true',
                                help='Generate detailed business insights report')
    
    # ----------------------------
    # Sample Data Command
    # ----------------------------
    sample_parser = subparsers.add_parser('sample', help='Generate sample VAHAN data for testing')
    sample_parser.add_argument('--output', '-o', default='sample_vahan_data.csv',
                               help='Output CSV filename for sample data')
    sample_parser.add_argument('--rows', type=int, default=None,
                               help='Number of rows for sample data (default full sample)')
    
    # ----------------------------
    # Global Options
    # ----------------------------
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')
    parser.add_argument('--log-file', default=None,
                        help='Log file path (auto-generated if omitted)')
    
    return parser

def scrape_command(args):
    """🚀 Execute scraping command with full validation and logging."""
    logger = get_logger(__name__)
    
    try:
        logger.info("🚀 Starting VAHAN data scraping...")
        
        # Initialize scraper
        scraper = VahanScraper()
        scraper.setup_driver(headless=args.headless)
        scraper.open_page()
        
        # Fetch available dropdown options
        logger.info("📋 Fetching available dropdown options...")
        dropdown_data = scraper.scrape_dropdowns()
        
        # Validate requested states and years
        available_states = dropdown_data.get('State', [])
        available_years = dropdown_data.get('Year', [])
        
        valid_states = [state for state in args.states if state in available_states]
        valid_years = [year for year in args.years if year in available_years]
        
        if not valid_states:
            logger.error(f"❌ No valid states found. Available: {available_states[:10]}")
            return False
        
        if not valid_years:
            logger.error(f"❌ No valid years found. Available: {available_years}")
            return False
        
        logger.info(f"✅ Scraping {len(valid_states)} states for {len(valid_years)} years")
        
        # Generate filter combinations
        from src.core.models import FilterCombination
        combinations = []
        for state in valid_states:
            for year in valid_years:
                if args.vehicle_types:
                    for vehicle_type in args.vehicle_types:
                        combinations.append(FilterCombination(state=state, year=year, vehicle_type=vehicle_type))
                else:
                    combinations.append(FilterCombination(state=state, year=year))
        
        # Scrape data
        scraped_data = scraper.scrape_multiple_combinations(combinations)
        
        if scraped_data.empty:
            logger.error("❌ No data was scraped")
            return False
        
        # Save scraped data
        output_file = args.output or Config.get_output_filename("scraped_vahan_data")
        saved_path = scraper.save_data(scraped_data, output_file)
        
        logger.info(f"✅ Scraping completed! Data saved to: {saved_path}")
        logger.info(f"📊 Total records scraped: {len(scraped_data)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        return False
    
    finally:
        if 'scraper' in locals():
            scraper.close()

def process_command(args):
    """🔄 Execute VAHAN data processing with full logging and export."""
    logger = get_logger(__name__)
    
    try:
        logger.info("🔄 Starting data processing...")

        # Initialize processor
        processor = VahanDataProcessor()
        
        # Load input data
        logger.info(f"📂 Loading input data from: {args.input}")
        data = processor.load_data(args.input)
        
        if data.empty:
            logger.warning("⚠️ Input data is empty. Nothing to process.")
            return False
        
        # Process all data (cleaning, transformations, calculations)
        logger.info("⚙️ Processing data...")
        result = processor.process_all(data)
        
        # Export processed data
        output_file = args.output or Config.get_output_filename("processed_vahan_data")
        saved_path = processor.export_processed_data(output_file)
        
        logger.info(f"✅ Processing completed! Data saved to: {saved_path}")
        logger.info(f"📊 Records processed: {result.records_processed:,}")
        logger.info(f"⏱️ Processing time: {result.processing_time:.2f} seconds")
        
        # Optionally export additional metrics
        if getattr(args, 'export_metrics', False):
            metrics_file = Config.get_output_filename("vahan_metrics")
            processor.export_metrics(metrics_file)
            logger.info(f"📈 Growth metrics exported to: {metrics_file}")
        
        return True

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return False

def analyze_command(args):
    """📈 Execute VAHAN data analysis with full reporting and insights."""
    logger = get_logger(__name__)
    
    try:
        logger.info("📈 Starting data analysis...")

        # Load processed data
        logger.info(f"📂 Loading processed data from: {args.input}")
        file_manager = FileManager()
        data = file_manager.load_dataframe(args.input)

        if data.empty:
            logger.warning("⚠️ Input data is empty. Analysis aborted.")
            return False

        # Initialize analyzers
        growth_analyzer = GrowthAnalyzer()
        insight_generator = InsightGenerator()

        # Perform growth analysis
        logger.info("📊 Calculating growth metrics...")
        cagr = growth_analyzer.calculate_compound_growth_rate(data)
        volatility = growth_analyzer.analyze_growth_volatility(data)
        patterns = growth_analyzer.identify_growth_patterns(data)

        # Generate insights if requested
        insights = None
        if getattr(args, 'generate_insights', False):
            logger.info("💡 Generating business insights...")
            # You can replace with real growth metrics if available
            growth_metrics = {
                'yoy_growth': {'2022-2023': 12.5, '2023-2024': 15.2},
                'category_growth': {'2W': {'2023-2024': 18.3}, '4W+': {'2023-2024': 8.7}},
                'state_growth': {'Karnataka': {'2023-2024': 22.1}}
            }
            insights = insight_generator.generate_market_insights(data, growth_metrics)
            logger.info("💡 Insights generation complete.")

        # Prepare output directory
        output_dir = Path(args.output_dir) if args.output_dir else Config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compile analysis report
        analysis_results = {
            'cagr': cagr,
            'volatility_analysis': volatility,
            'growth_patterns': patterns
        }
        if insights:
            analysis_results['business_insights'] = insights

        # Save report as JSON
        import json
        report_file = output_dir / Config.get_output_filename("analysis_report", "json")
        with open(report_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        logger.info(f"✅ Analysis completed! Report saved to: {report_file}")
        logger.info(f"📈 CAGR: {cagr:.2f}%")
        logger.info(f"📊 Volatility metrics: {volatility}")
        
        return True

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False

def sample_command(args):
    """🧪 Generate and save VAHAN sample data for testing purposes."""
    logger = get_logger(__name__)

    try:
        logger.info("🧪 Starting sample data generation...")

        # Generate sample data
        sample_data = create_sample_data(num_rows=getattr(args, 'rows', None))

        if sample_data.empty:
            logger.warning("⚠️ Generated sample data is empty. Nothing to save.")
            return False

        # Save sample data
        file_manager = FileManager()
        output_file = args.output or Config.get_output_filename("sample_vahan_data", "csv")
        saved_path = file_manager.save_dataframe(sample_data, output_file)

        logger.info(f"✅ Sample data generated successfully! Saved to: {saved_path}")
        logger.info(f"📊 Total sample records: {len(sample_data):,}")

        return True

    except Exception as e:
        logger.error(f"❌ Sample generation failed: {e}")
        return False

def main():
    """🚀 Main CLI Entry Point for VAHAN Vehicle Registration Data Scraper & Analyzer."""
    # ----------------------------
    # 1️⃣ Setup CLI and Parse Args
    # ----------------------------
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # ----------------------------
    # 2️⃣ Setup Logging
    # ----------------------------
    setup_logging(args.log_level, args.log_file)
    logger = get_logger(__name__)
    logger.info("🚗 VAHAN CLI Initialized")
    
    # ----------------------------
    # 3️⃣ Ensure Required Directories Exist
    # ----------------------------
    Config.ensure_directories()
    logger.debug("📁 Ensured output & temp directories exist")
    
    # ----------------------------
    # 4️⃣ Execute CLI Command
    # ----------------------------
    try:
        if args.command == 'scrape':
            logger.info("🌐 Running scraping command...")
            success = scrape_command(args)

        elif args.command == 'process':
            logger.info("🔄 Running processing command...")
            success = process_command(args)

        elif args.command == 'analyze':
            logger.info("📊 Running analysis command...")
            success = analyze_command(args)

        elif args.command == 'sample':
            logger.info("🧪 Running sample data generation command...")
            success = sample_command(args)

        else:
            logger.warning("⚠️ No valid command provided. Showing help...")
            parser.print_help()
            success = False
        
        # ----------------------------
        # 5️⃣ Exit Status
        # ----------------------------
        if success:
            logger.info("✅ Operation completed successfully! 🎉")
            sys.exit(0)
        else:
            logger.error("❌ Operation failed! ❗")
            sys.exit(1)
    
    # ----------------------------
    # 6️⃣ Handle Interrupts & Errors
    # ----------------------------
    except KeyboardInterrupt:
        logger.info("⏹️ Operation cancelled by user via keyboard interrupt")
        sys.exit(1)
    except VahanScraperError as e:
        logger.error(f"❌ VAHAN Scraper Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error occurred: {e}")
        sys.exit(1)

# ----------------------------
# 7️⃣ Entrypoint
# ----------------------------
if __name__ == "__main__":
    main()
