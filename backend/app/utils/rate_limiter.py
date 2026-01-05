"""
Rate limiting utility to prevent excessive API usage.
Tracks combined Firecrawl + AWS calls with a total limit.
"""
import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from threading import Lock
from collections import defaultdict

from app.config import Config

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter to control API call frequency and total usage.
    Tracks calls per service and enforces a combined total limit.
    """
    
    def __init__(self, max_total_calls: Optional[int] = None):
        """
        Initialize rate limiter with call tracking.
        
        Args:
            max_total_calls: Maximum total calls across all services (defaults to Config.MAX_TOTAL_CALLS)
        """
        self.max_total_calls = max_total_calls or Config.MAX_TOTAL_CALLS
        self.call_history: Dict[str, list] = defaultdict(list)  # service -> list of timestamps
        self.total_calls: Dict[str, int] = defaultdict(int)  # service -> total count
        self.lock = Lock()
        self.start_time = datetime.now()
    
    def can_make_call(self, service: str) -> Tuple[bool, str]:
        """
        Check if a call can be made based on total rate limits.
        
        Args:
            service: Service name (e.g., 'firecrawl', 'textract')
        
        Returns:
            Tuple of (can_call: bool, reason: str)
        """
        with self.lock:
            # Calculate total calls across all services
            total_calls_made = sum(self.total_calls.values())
            
            # Check total calls limit (combined across all services)
            if total_calls_made >= self.max_total_calls:
                return False, (
                    f"Total call limit reached: {total_calls_made}/{self.max_total_calls} "
                    f"calls across all services"
                )
            
            return True, "OK"
    
    def record_call(self, service: str):
        """Record that a call was made."""
        with self.lock:
            now = datetime.now()
            self.call_history[service].append(now)
            self.total_calls[service] += 1
            
            total_calls_made = sum(self.total_calls.values())
            logger.info(
                f"Recorded call for {service}. "
                f"Total calls: {total_calls_made}/{self.max_total_calls}"
            )
    
    def get_stats(self, service: Optional[str] = None) -> Dict:
        """
        Get statistics for a service or all services.
        
        Args:
            service: Optional service name. If None, returns combined stats.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            now = datetime.now()
            
            if service:
                # Get stats for specific service
                recent_calls = self.call_history[service]
                
                # Remove calls older than 24 hours
                cutoff_24h = now - timedelta(hours=24)
                recent_calls[:] = [ts for ts in recent_calls if ts > cutoff_24h]
                
                cutoff_1m = now - timedelta(minutes=1)
                cutoff_1h = now - timedelta(hours=1)
                
                return {
                    'service': service,
                    'total_calls': self.total_calls[service],
                    'calls_last_minute': sum(1 for ts in recent_calls if ts > cutoff_1m),
                    'calls_last_hour': sum(1 for ts in recent_calls if ts > cutoff_1h),
                    'calls_last_day': len(recent_calls),
                }
            else:
                # Get combined stats
                total_calls_made = sum(self.total_calls.values())
                remaining_calls = max(0, self.max_total_calls - total_calls_made)
                
                calls_by_service = dict(self.total_calls)
                
                return {
                    'total_calls': total_calls_made,
                    'max_calls': self.max_total_calls,
                    'remaining_calls': remaining_calls,
                    'calls_by_service': calls_by_service,
                    'session_duration': (now - self.start_time).total_seconds()
                }
    
    def wait_if_needed(self, service: str, min_delay_seconds: float = 1.0):
        """
        Wait if needed to respect rate limits.
        
        Args:
            service: Service name
            min_delay_seconds: Minimum delay between calls
        """
        if min_delay_seconds > 0:
            time.sleep(min_delay_seconds)
    
    def reset(self):
        """Reset all call tracking (useful for testing)."""
        with self.lock:
            self.call_history.clear()
            self.total_calls.clear()
            self.start_time = datetime.now()
            logger.info("Rate limiter reset")




