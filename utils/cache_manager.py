# utils/cache_manager.py
"""
Cache manager for storing API responses and reducing redundant calls
Uses file-based caching for simplicity
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional
import hashlib
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages cached data to reduce API calls"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_index = self._load_cache_index()
        
        # Clean up old cache on initialization
        self._cleanup_old_cache()
    
    def _load_cache_index(self) -> dict:
        """Load cache index from file"""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save cache index to file"""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache filename from key"""
        # Create hash of key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        
        cache_key = self._get_cache_key(key)
        
        # Check if key exists in index
        if key not in self.cache_index:
            return None
        
        # Check if cache has expired
        expiry = datetime.fromisoformat(self.cache_index[key]['expiry'])
        if datetime.now() > expiry:
            self.delete(key)
            return None
        
        # Load cached data
        cache_file = os.path.join(self.cache_dir, cache_key)
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                logger.debug(f"Cache hit for key: {key}")
                return data
        except Exception as e:
            logger.error(f"Error loading cache for {key}: {str(e)}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, expiry_minutes: int = 15):
        """Set value in cache with expiry time"""
        
        cache_key = self._get_cache_key(key)
        cache_file = os.path.join(self.cache_dir, cache_key)
        
        # Calculate expiry time
        expiry = datetime.now() + timedelta(minutes=expiry_minutes)
        
        # Save to cache file
        try:
            with open(cache_file, 'w') as f:
                json.dump(value, f, default=self._json_serializer, indent=2)
            
            # Update index
            self.cache_index[key] = {
                'file': cache_key,
                'expiry': expiry.isoformat(),
                'created': datetime.now().isoformat()
            }
            self._save_cache_index()
            
            logger.debug(f"Cached data for key: {key}, expires in {expiry_minutes} minutes")
            
        except Exception as e:
            logger.error(f"Error caching data for {key}: {str(e)}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for pandas and numpy objects"""
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dict and ensure index is serializable
            df_dict = obj.to_dict()
            # Convert any Timestamp indexes to strings
            for col in df_dict:
                if isinstance(df_dict[col], dict):
                    df_dict[col] = {
                        str(k) if isinstance(k, (pd.Timestamp, datetime)) else k: v
                        for k, v in df_dict[col].items()
                    }
            return df_dict
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    def delete(self, key: str):
        """Delete item from cache"""
        
        if key in self.cache_index:
            cache_key = self.cache_index[key]['file']
            cache_file = os.path.join(self.cache_dir, cache_key)
            
            # Remove file
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except:
                    pass
            
            # Remove from index
            del self.cache_index[key]
            self._save_cache_index()
    
    def clear_all(self):
        """Clear all cached data"""
        
        # Remove all cache files
        for key, info in self.cache_index.items():
            cache_file = os.path.join(self.cache_dir, info['file'])
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        # Clear index
        self.cache_index = {}
        self._save_cache_index()
        
        logger.info("Cache cleared")
    
    def _cleanup_old_cache(self):
        """Remove expired cache entries"""
        
        expired_keys = []
        
        for key, info in self.cache_index.items():
            try:
                expiry = datetime.fromisoformat(info['expiry'])
                if datetime.now() > expiry:
                    expired_keys.append(key)
            except:
                expired_keys.append(key)
        
        # Delete expired entries
        for key in expired_keys:
            self.delete(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        
        total_size = 0
        file_count = 0
        
        for key, info in self.cache_index.items():
            cache_file = os.path.join(self.cache_dir, info['file'])
            if os.path.exists(cache_file):
                total_size += os.path.getsize(cache_file)
                file_count += 1
        
        return {
            'total_entries': len(self.cache_index),
            'total_files': file_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': self.cache_dir
        }
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache and is not expired"""
        return self.get(key) is not None