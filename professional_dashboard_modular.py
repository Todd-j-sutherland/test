"""
Simplified Professional Dashboard Entry Point
Uses the new modular dashboard structure for better maintainability
"""

import os
import sys
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Main entry point for the professional dashboard"""
    try:
        # Import the modular dashboard
        from dashboard.main_dashboard import ProfessionalDashboard
        
        # Create and run the dashboard
        dashboard = ProfessionalDashboard()
        dashboard.run()
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install streamlit plotly pandas numpy")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        logging.error(f"Dashboard startup error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
