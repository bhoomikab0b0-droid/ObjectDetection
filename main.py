"""
Main entry point for the Blind Navigation System
"""

from blind_navigation import BlindNavigationSystem

def main():
    """Initialize and run the blind navigation system"""
    print("Starting Blind Navigation System...")
    navigation_system = BlindNavigationSystem()
    navigation_system.run()

if __name__ == "__main__":
    main()