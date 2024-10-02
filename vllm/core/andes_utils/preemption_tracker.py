
class PreemptionTracker:
    def __init__(self, window_size=600, bucket_duration=10):
        """Initialize the request tracker.
        
        Args:
        window_size: The total time window in seconds (default: 600 seconds or 10 minutes).
        bucket_duration: Duration of each bucket in seconds (default: 10 seconds).
        """
        self.window_size = window_size
        self.bucket_duration = bucket_duration
        self.num_buckets = window_size // bucket_duration
        self.buckets = [0] * self.num_buckets  # Each bucket stores the number of requests in that period
        self.timestamps = [0] * self.num_buckets  # Each bucket also stores the last update timestamp
        self.total_requests = 0
        self.bucket_duration_seconds = bucket_duration
    
    def _get_current_bucket_index(self, current_time):
        """Returns the current bucket index based on time."""
        return int(current_time // self.bucket_duration) % self.num_buckets
    
    def _prune_old_bucket(self, bucket_index, current_time):
        """Reset the bucket if it's older than the current window size. Triggered only when adding requests."""
        if current_time - self.timestamps[bucket_index] >= self.window_size:
            self.total_requests -= self.buckets[bucket_index]
            self.buckets[bucket_index] = 0  # Clear out old bucket
            self.timestamps[bucket_index] = current_time
    
    def add_request(self, timestamp, count = 1):
        """Adds a new request to the current bucket."""
        bucket_index = self._get_current_bucket_index(timestamp)
        self._prune_old_bucket(bucket_index, timestamp)
        self.buckets[bucket_index] += count
        self.total_requests += count
    
    def get_request_count(self):
        """Returns the total number of requests in the last window_size seconds."""
        return self.total_requests


# import time

# tracker = PreemptionTracker(10, 1)

# # Simulate requests arriving at different times
# tracker.add_request(time.time(),5)  # First request
# tracker.add_request(time.time())  # Second request
# tracker.add_request(time.time())  # Second request
# time.sleep(10)          

# print(f"Requests in the past 0.5 minutes: {tracker.get_request_count()}")  # Should print 2

# time.sleep(5)         
# tracker.add_request(time.time())  # Third request

# print(f"Requests in the past 0.5 minutes: {tracker.get_request_count()}")  # Should print 3

# time.sleep(5)         
# tracker.add_request(time.time())  # New request

# print(f"Requests in the past 0.5 minutes: {tracker.get_request_count()}")  # Should print 1 (only the latest request)


# time.sleep(25)        
# tracker.add_request(time.time())  # New request

# print(f"Requests in the past 0.5 minutes: {tracker.get_request_count()}")  # Should print 1 (only the latest request)



# time.sleep(20)        # Wait for over 10 minutes (so older buckets expire)
# print(f"Requests in the past 0.5 minutes: {tracker.get_request_count()}")  # Should print 1 (only the latest request)



