"""
Try datagovindia package - sync metadata first then search.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import datagovindia
    print("datagovindia package imported successfully")
    
    # Try sync_metadata
    print("\nSyncing metadata...")
    try:
        datagovindia.sync_metadata()
        print("Metadata synced!")
    except Exception as e:
        print(f"sync_metadata error: {e}")
    
    # Try search
    print("\nSearching for crop survey data...")
    try:
        results = datagovindia.search("crop survey dakshina kannada")
        print(f"Search results: {results}")
    except Exception as e:
        print(f"Search error: {e}")
        
except ImportError as e:
    print(f"Import error: {e}")
    
except Exception as e:
    print(f"Error: {e}")
