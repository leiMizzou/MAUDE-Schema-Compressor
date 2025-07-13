"""
Similarity calculation module for MAUDE Schema Compressor
Handles API-based semantic similarity calculations with caching and optimization
"""
import json
import logging
import time
import requests
import re
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import config

class SimilarityCache:
    """Manages similarity score caching"""
    
    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or config.paths.cache_file
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        self.load_cache()
    
    def load_cache(self):
        """Load similarity cache from file"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            self.logger.info(f"Loaded {len(self.cache)} cached similarity scores")
        except FileNotFoundError:
            self.logger.info("No existing cache file found, starting with empty cache")
            self.cache = {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Error loading cache file: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save similarity cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Saved {len(self.cache)} similarity scores to cache")
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")
    
    def get_cache_key(self, table1: str, table2: str) -> str:
        """Generate cache key for table pair"""
        return "|".join(sorted([table1, table2]))
    
    def get_similarity(self, table1: str, table2: str) -> Optional[float]:
        """Get similarity score from cache"""
        key = self.get_cache_key(table1, table2)
        return self.cache.get(key)
    
    def set_similarity(self, table1: str, table2: str, score: float):
        """Set similarity score in cache"""
        key = self.get_cache_key(table1, table2)
        self.cache[key] = score
    
    def cache_size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

class APIClient:
    """Handles API communication for similarity calculations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = config.api.deepseek_api_key
        self.base_url = config.api.deepseek_base_url
        self.retry_limit = config.api.retry_limit
        self.timeout = config.api.timeout
        
        if not self.api_key:
            self.logger.warning("DEEPSEEK_API_KEY not provided. API calls will fail.")
            # Don't raise error, allow initialization for cache-only usage
    
    def call_api(self, prompt: str) -> str:
        """
        Make API call to calculate similarity
        Returns the response content
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        for attempt in range(1, self.retry_limit + 1):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                elapsed_time = time.time() - start_time
                
                self.logger.debug(f"API call attempt {attempt} took {elapsed_time:.2f}s")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    
                    # Basic JSON integrity check
                    if content.count('{') == content.count('}'):
                        return content
                    else:
                        self.logger.warning(f"Incomplete JSON response on attempt {attempt}")
                else:
                    self.logger.error(f"API error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request exception on attempt {attempt}: {e}")
            
            if attempt < self.retry_limit:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        self.logger.error(f"All {self.retry_limit} API call attempts failed")
        return ""
    
    def extract_similarity_score(self, response: str) -> Optional[float]:
        """
        Extract similarity score from API response
        Returns similarity score or None if extraction fails
        """
        try:
            # Remove code block markers
            text = re.sub(r'```json', '', response)
            text = re.sub(r'```', '', text)
            
            # Find JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                self.logger.error("No JSON object found in response")
                return None
            
            json_str = json_match.group()
            json_obj = json.loads(json_str)
            
            # Extract similarity score
            if "similarity_score" in json_obj:
                score = float(json_obj["similarity_score"])
                if 0.0 <= score <= 1.0:
                    return score
                else:
                    self.logger.warning(f"Similarity score {score} out of range [0,1]")
                    return max(0.0, min(1.0, score))  # Clamp to valid range
            else:
                self.logger.error("'similarity_score' not found in response")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error: {e}")
            return None
        except ValueError as e:
            self.logger.error(f"Error converting similarity score to float: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error extracting similarity score: {e}")
            return None

class SimilarityCalculator:
    """Main class for calculating semantic similarity between tables"""
    
    def __init__(self, update_cache: bool = True):
        self.logger = logging.getLogger(__name__)
        self.cache = SimilarityCache()
        self.update_cache = update_cache  # Control whether to update cache with new calculations
        
        # Initialize API client only if API key is available
        try:
            self.api_client = APIClient()
            self.api_available = True
        except Exception as e:
            self.logger.warning(f"API client not available: {e}")
            self.api_client = None
            self.api_available = False
    
    def create_similarity_prompt(self, table1_desc: str, table2_desc: str) -> str:
        """
        Create prompt for similarity calculation
        """
        prompt = f"""
Please evaluate the similarity between the following two database tables. Return a similarity score between 0 and 1, where 0 indicates completely dissimilar and 1 indicates identical similarity.

Consider the following aspects when evaluating similarity:
1. Field names and their semantic meanings
2. Data types and structure
3. Sample data patterns
4. Business domain and purpose

Table 1 Description: {table1_desc}

Table 2 Description: {table2_desc}

Please return only a JSON object in the format {{"similarity_score": 0.85}} without any additional content.
"""
        return prompt
    
    def calculate_pair_similarity(self, table1: str, table2: str, 
                                descriptions: Dict[str, str]) -> float:
        """
        Calculate similarity score for a single table pair
        """
        # Check cache first
        cached_score = self.cache.get_similarity(table1, table2)
        if cached_score is not None:
            self.logger.debug(f"Using cached similarity for ({table1}, {table2}): {cached_score}")
            return cached_score
        
        # Get descriptions
        desc1 = descriptions.get(table1, "")
        desc2 = descriptions.get(table2, "")
        
        if not desc1 or not desc2:
            self.logger.warning(f"Missing description for table pair ({table1}, {table2})")
            return 0.0
        
        # Try to call API if available
        if self.api_available and self.api_client:
            prompt = self.create_similarity_prompt(desc1, desc2)
            response = self.api_client.call_api(prompt)
            
            if response:
                score = self.api_client.extract_similarity_score(response)
                if score is not None:
                    # Cache the result only if update_cache is True
                    if self.update_cache:
                        self.cache.set_similarity(table1, table2, score)
                        self.logger.debug(f"Calculated similarity for ({table1}, {table2}): {score} (cached)")
                    else:
                        self.logger.debug(f"Calculated similarity for ({table1}, {table2}): {score} (not cached)")
                    return score
        
        # Fallback: return 0.0 if no API or calculation failed
        if not self.api_available:
            self.logger.debug(f"No cached similarity for ({table1}, {table2}) and API not available, using 0.0")
        else:
            self.logger.warning(f"Failed to calculate similarity for ({table1}, {table2}), using 0.0")
        
        # Cache fallback value only if update_cache is True
        if self.update_cache:
            self.cache.set_similarity(table1, table2, 0.0)
        return 0.0
    
    def calculate_batch_similarity(self, pairs: List[Tuple[str, str]], 
                                 descriptions: Dict[str, str],
                                 max_workers: int = 3) -> Dict[Tuple[str, str], float]:
        """
        Calculate similarity scores for multiple table pairs using parallel processing
        """
        similarity_scores = {}
        
        # Filter pairs that are not already cached
        uncached_pairs = []
        for pair in pairs:
            table1, table2 = pair
            cached_score = self.cache.get_similarity(table1, table2)
            if cached_score is not None:
                similarity_scores[pair] = cached_score
            else:
                uncached_pairs.append(pair)
        
        self.logger.info(f"Processing {len(uncached_pairs)} uncached pairs "
                        f"({len(pairs) - len(uncached_pairs)} from cache)")
        
        if not uncached_pairs:
            return similarity_scores
        
        def calculate_single_pair(pair: Tuple[str, str]) -> Tuple[Tuple[str, str], float]:
            table1, table2 = pair
            score = self.calculate_pair_similarity(table1, table2, descriptions)
            return pair, score
        
        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(calculate_single_pair, pair): pair 
                for pair in uncached_pairs
            }
            
            with tqdm(total=len(uncached_pairs), desc="Calculating similarities") as pbar:
                for future in as_completed(future_to_pair):
                    try:
                        pair, score = future.result()
                        similarity_scores[pair] = score
                        pbar.update(1)
                    except Exception as e:
                        pair = future_to_pair[future]
                        self.logger.error(f"Error calculating similarity for {pair}: {e}")
                        similarity_scores[pair] = 0.0
                        pbar.update(1)
        
        # Save cache after batch processing only if update_cache is True
        if self.update_cache:
            self.cache.save_cache()
        
        return similarity_scores
    
    def calculate_cluster_similarities(self, clusters: List[List[str]], 
                                     descriptions: Dict[str, str],
                                     prefilter_threshold: float = None,
                                     max_workers: int = 3) -> Tuple[Dict[Tuple[str, str], float], int]:
        """
        Calculate similarities within clusters with optional pre-filtering
        Returns similarity scores and number of pairs processed
        """
        if prefilter_threshold is None:
            prefilter_threshold = config.clustering.prefilter_jaccard_threshold
        
        # Generate pairs within clusters
        all_pairs = []
        total_pairs = 0
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            n = len(cluster)
            for i in range(n):
                for j in range(i + 1, n):
                    total_pairs += 1
                    pair = tuple(sorted((cluster[i], cluster[j])))
                    all_pairs.append(pair)
        
        self.logger.info(f"Generated {len(all_pairs)} table pairs from {len(clusters)} clusters")
        
        # Apply pre-filtering if enabled
        if prefilter_threshold > 0:
            filtered_pairs = self._prefilter_pairs(all_pairs, descriptions, prefilter_threshold)
            self.logger.info(f"Pre-filtering reduced pairs from {len(all_pairs)} to {len(filtered_pairs)}")
        else:
            filtered_pairs = all_pairs
        
        # Calculate similarities
        similarity_scores = self.calculate_batch_similarity(filtered_pairs, descriptions, max_workers)
        
        return similarity_scores, len(filtered_pairs)
    
    def _prefilter_pairs(self, pairs: List[Tuple[str, str]], 
                        descriptions: Dict[str, str],
                        threshold: float) -> List[Tuple[str, str]]:
        """
        Pre-filter pairs using Jaccard similarity of field names
        """
        filtered_pairs = []
        
        for table1, table2 in pairs:
            try:
                # Extract field names from descriptions
                fields1 = self._extract_fields_from_description(descriptions.get(table1, ""))
                fields2 = self._extract_fields_from_description(descriptions.get(table2, ""))
                
                # Calculate Jaccard similarity
                intersection = fields1.intersection(fields2)
                union = fields1.union(fields2)
                jaccard = len(intersection) / len(union) if union else 0.0
                
                if jaccard >= threshold:
                    filtered_pairs.append((table1, table2))
                    
            except Exception as e:
                self.logger.warning(f"Error pre-filtering pair ({table1}, {table2}): {e}")
                # Include pair if filtering fails
                filtered_pairs.append((table1, table2))
        
        return filtered_pairs
    
    def _extract_fields_from_description(self, description: str) -> set:
        """
        Extract field names from table description
        """
        import re
        
        # Look for pattern "field_name (type)"
        pattern = r'(\w+)\s*\([^)]+\)'
        matches = re.findall(pattern, description)
        return set(match.strip().lower() for match in matches)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_size': self.cache.cache_size(),
            'cache_file': self.cache.cache_file
        }