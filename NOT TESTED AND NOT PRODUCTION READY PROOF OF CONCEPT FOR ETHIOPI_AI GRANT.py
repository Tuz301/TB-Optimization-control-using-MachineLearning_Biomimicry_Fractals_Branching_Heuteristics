"""
Production-Ready Ethiopia TB AI Cost-Effectiveness Analysis System
"""

import os
import sys
import logging
import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple, Protocol
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator, Field
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tb_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class SystemConfig:
    """Centralized configuration management with environment variable support."""
    
    # Database Configuration
    HMIS_CONNECTION_STRING: str = field(default_factory=lambda: os.getenv('HMIS_DB_URL', 'sqlite:///tb_data.db'))
    WHO_API_KEY: str = field(default_factory=lambda: os.getenv('WHO_API_KEY', ''))
    REDIS_URL: str = field(default_factory=lambda: os.getenv('REDIS_URL', 'redis://localhost:6379'))
    
    # Model Parameters
    MCMC_SAMPLES: int = field(default_factory=lambda: int(os.getenv('MCMC_SAMPLES', '2000')))
    MCMC_TUNE: int = field(default_factory=lambda: int(os.getenv('MCMC_TUNE', '1000')))
    CONVERGENCE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv('CONVERGENCE_THRESHOLD', '1.05')))
    
    # Economic Parameters
    WTP_THRESHOLD: float = field(default_factory=lambda: float(os.getenv('WTP_THRESHOLD', '500')))
    DISCOUNT_RATE: float = field(default_factory=lambda: float(os.getenv('DISCOUNT_RATE', '0.03')))
    TIME_HORIZON: int = field(default_factory=lambda: int(os.getenv('TIME_HORIZON', '10')))
    
    # Optimization Parameters
    DEFAULT_BUDGET: float = field(default_factory=lambda: float(os.getenv('DEFAULT_BUDGET', '10000000')))
    MAX_OPTIMIZATION_TIME: int = field(default_factory=lambda: int(os.getenv('MAX_OPT_TIME', '300')))
    EQUITY_WEIGHT: float = field(default_factory=lambda: float(os.getenv('EQUITY_WEIGHT', '0.3')))
    
    # System Parameters
    CACHE_TTL: int = field(default_factory=lambda: int(os.getenv('CACHE_TTL', '3600')))
    MAX_CONCURRENT_JOBS: int = field(default_factory=lambda: int(os.getenv('MAX_CONCURRENT_JOBS', '4')))
    MODEL_VERSION_RETENTION: int = field(default_factory=lambda: int(os.getenv('MODEL_VERSION_RETENTION', '10')))
    
    # Validation Parameters
    MIN_FACILITIES: int = 50
    MAX_MISSING_DATA_RATE: float = 0.2
    MIN_TIME_PERIODS: int = 12
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.EQUITY_WEIGHT < 0 or self.EQUITY_WEIGHT > 1:
            raise ValueError("EQUITY_WEIGHT must be between 0 and 1")
        if self.DISCOUNT_RATE < 0 or self.DISCOUNT_RATE > 0.2:
            raise ValueError("DISCOUNT_RATE must be between 0 and 0.2")


# =============================================================================
# DATA MODELS AND VALIDATION
# =============================================================================

class FacilityData(BaseModel):
    """Pydantic model for facility data validation."""
    
    facility_id: int = Field(..., ge=0)
    period: int = Field(..., ge=0)
    facility_type: str = Field(..., regex='^(Hospital|Health Center|Health Post)$')
    region: str = Field(..., min_length=1)
    urban_rural: str = Field(..., regex='^(Urban|Rural)$')
    staff_count: int = Field(..., ge=0, le=1000)
    cases_detected: int = Field(..., ge=0, le=10000)
    treatment_success_rate: float = Field(..., ge=0.0, le=1.0)
    population_density: float = Field(..., ge=0.0)
    travel_time_hours: float = Field(..., ge=0.0, le=24.0)
    has_gene_xpert: bool
    has_mobile_xray: bool
    
    @validator('cases_detected')
    def validate_cases_realistic(cls, v, values):
        """Ensure cases detected is realistic given facility capacity."""
        if 'staff_count' in values and values['staff_count'] > 0:
            max_cases = values['staff_count'] * 50  # Rough capacity estimate
            if v > max_cases:
                logger.warning(f"Cases detected ({v}) exceeds expected capacity ({max_cases})")
        return v


class OptimizationRequest(BaseModel):
    """API request model for resource optimization."""
    
    budget: float = Field(..., gt=0, le=100000000)
    priority_regions: List[str] = Field(default=[])
    equity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    intervention_types: List[str] = Field(default=['gene_xpert', 'mobile_xray', 'training'])
    time_horizon: int = Field(default=5, ge=1, le=20)


class AllocationResult(BaseModel):
    """Result model for resource allocation."""
    
    allocation_id: str
    total_cost: float
    facilities_funded: int
    expected_cases_detected: float
    expected_dalys_averted: float
    icer: float
    equity_score: float
    confidence_interval: Tuple[float, float]
    interventions: List[Dict[str, Any]]


# =============================================================================
# CORE INTERFACES AND PROTOCOLS
# =============================================================================

class DataConnector(Protocol):
    """Protocol for data connectors."""
    
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data for the specified date range."""
        ...
    
    def validate_connection(self) -> bool:
        """Validate connection to data source."""
        ...


class ModelInterface(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction uncertainty."""
        pass


# =============================================================================
# DATA MANAGEMENT LAYER
# =============================================================================

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class CacheManager:
    """Redis-based cache manager with fallback to in-memory cache."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        try:
            self.redis_client = redis.from_url(config.REDIS_URL)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory cache: {e}")
            self.use_redis = False
            self.memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.use_redis:
                value = self.redis_client.get(key)
                return joblib.loads(value) if value else None
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.config.CACHE_TTL
            if self.use_redis:
                serialized = joblib.dumps(value)
                return self.redis_client.setex(key, ttl, serialized)
            else:
                self.memory_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False


class DataManager:
    """Centralized data management with validation, caching, and multiple source integration."""
    
    def __init__(self, config: SystemConfig, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self.data_connectors: Dict[str, DataConnector] = {}
        self._last_validation_report = None
    
    def register_connector(self, name: str, connector: DataConnector) -> None:
        """Register a data connector."""
        if not connector.validate_connection():
            raise ConnectionError(f"Cannot connect to {name} data source")
        self.data_connectors[name] = connector
        logger.info(f"Registered data connector: {name}")
    
    async def load_facility_data(self, 
                               start_date: datetime,
                               end_date: datetime,
                               force_refresh: bool = False) -> pd.DataFrame:
        """Load and validate facility data from multiple sources."""
        cache_key = f"facility_data_{start_date.isoformat()}_{end_date.isoformat()}"
        
        # Try cache first
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info("Retrieved facility data from cache")
                return cached_data
        
        # Fetch from multiple sources
        dataframes = []
        for name, connector in self.data_connectors.items():
            try:
                df = await connector.fetch_data(start_date, end_date)
                df['data_source'] = name
                dataframes.append(df)
                logger.info(f"Fetched {len(df)} records from {name}")
            except Exception as e:
                logger.error(f"Failed to fetch from {name}: {e}")
        
        if not dataframes:
            raise DataValidationError("No data sources available")
        
        # Merge and deduplicate
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = self._merge_duplicate_facilities(combined_df)
        
        # Validate data
        validated_df = self._validate_and_clean_data(combined_df)
        
        # Cache the result
        self.cache.set(cache_key, validated_df)
        
        return validated_df
    
    def _merge_duplicate_facilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicate facility records with conflict resolution."""
        # Priority order for data sources
        source_priority = {'hmis': 1, 'who': 2, 'manual': 3}
        
        df['priority'] = df['data_source'].map(source_priority).fillna(99)
        
        # Keep the highest priority record for each facility-period
        df_sorted = df.sort_values(['facility_id', 'period', 'priority'])
        df_dedup = df_sorted.groupby(['facility_id', 'period']).first().reset_index()
        
        logger.info(f"Merged {len(df)} records into {len(df_dedup)} unique facility-periods")
        return df_dedup.drop(['data_source', 'priority'], axis=1)
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning."""
        original_count = len(df)
        validation_errors = []
        
        # Basic structure validation
        required_columns = ['facility_id', 'period', 'cases_detected', 'staff_count']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
        
        # Row-by-row validation using Pydantic
        valid_rows = []
        for idx, row in df.iterrows():
            try:
                facility_data = FacilityData(**row.to_dict())
                valid_rows.append(row)
            except Exception as e:
                validation_errors.append(f"Row {idx}: {str(e)}")
        
        if not valid_rows:
            raise DataValidationError("No valid data rows found")
        
        validated_df = pd.DataFrame(valid_rows)
        
        # Data quality checks
        missing_rate = validated_df.isnull().sum().sum() / (len(validated_df) * len(validated_df.columns))
        if missing_rate > self.config.MAX_MISSING_DATA_RATE:
            raise DataValidationError(f"Missing data rate {missing_rate:.2%} exceeds threshold")
        
        if len(validated_df) < self.config.MIN_FACILITIES:
            raise DataValidationError(f"Insufficient facilities: {len(validated_df)} < {self.config.MIN_FACILITIES}")
        
        # Store validation report
        self._last_validation_report = {
            'original_rows': original_count,
            'valid_rows': len(validated_df),
            'validation_errors': validation_errors[:10],  # First 10 errors
            'missing_data_rate': missing_rate,
            'validation_timestamp': datetime.now()
        }
        
        logger.info(f"Data validation complete: {len(validated_df)}/{original_count} rows valid")
        return validated_df
    
    def get_validation_report(self) -> Optional[Dict]:
        """Get the last data validation report."""
        return self._last_validation_report


# =============================================================================
# MODELING LAYER
# =============================================================================

class BayesianHierarchicalModel(ModelInterface):
    """Production-ready Bayesian hierarchical model with convergence diagnostics."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = None
        self.trace = None
        self.fitted = False
        self.convergence_diagnostics = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Bayesian hierarchical model with proper error handling."""
        try:
            import pymc3 as pm
            import theano.tensor as tt
            
            # Prepare data
            region_ids, region_map = pd.factorize(X['region'])
            n_regions = len(region_map)
            n_features = X.select_dtypes(include=[np.number]).shape[1]
            
            with pm.Model() as hierarchical_model:
                # Hyperpriors
                mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
                sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=5)
                
                # Region random effects
                alpha_region = pm.Normal('alpha_region', mu=mu_alpha, 
                                       sigma=sigma_alpha, shape=n_regions)
                
                # Fixed effects with regularization
                beta = pm.Laplace('beta', mu=0, b=1, shape=n_features)
                
                # Model
                X_numeric = X.select_dtypes(include=[np.number]).values
                mu = alpha_region[region_ids] + pm.math.dot(X_numeric, beta)
                
                # Overdispersion
                sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
                
                # Likelihood
                y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.exp(mu), 
                                         alpha=sigma_obs, observed=y.values)
                
                # Sample with multiple chains
                self.trace = pm.sample(
                    draws=self.config.MCMC_SAMPLES,
                    tune=self.config.MCMC_TUNE,
                    chains=4,
                    cores=4,
                    target_accept=0.95,
                    return_inferencedata=True
                )
            
            self.model = hierarchical_model
            self._check_convergence()
            self.fitted = True
            
            logger.info("Bayesian model fitted successfully")
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise RuntimeError(f"Bayesian model fitting failed: {e}")
    
    def _check_convergence(self) -> None:
        """Check MCMC convergence and log diagnostics."""
        try:
            import arviz as az
            
            # R-hat convergence diagnostic
            rhat = az.rhat(self.trace)
            max_rhat = float(rhat.max().values)
            
            # Effective sample size
            ess = az.ess(self.trace)
            min_ess = float(ess.min().values)
            
            # Energy diagnostics
            energy = az.bfmi(self.trace)
            min_energy = float(energy.min())
            
            self.convergence_diagnostics = {
                'max_rhat': max_rhat,
                'min_ess': min_ess,
                'min_energy': min_energy,
                'converged': max_rhat < self.config.CONVERGENCE_THRESHOLD
            }
            
            if not self.convergence_diagnostics['converged']:
                logger.warning(f"Model may not have converged: R-hat = {max_rhat}")
            
        except Exception as e:
            logger.error(f"Convergence check failed: {e}")
            self.convergence_diagnostics = {'error': str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with error handling."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        try:
            import pymc3 as pm
            
            with self.model:
                # Posterior predictive sampling
                ppc = pm.sample_posterior_predictive(self.trace, samples=500)
            
            return ppc['y_obs'].mean(axis=0)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction uncertainty intervals."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before uncertainty estimation")
        
        try:
            import pymc3 as pm
            
            with self.model:
                ppc = pm.sample_posterior_predictive(self.trace, samples=500)
            
            # Return 95% credible interval width
            lower = np.percentile(ppc['y_obs'], 2.5, axis=0)
            upper = np.percentile(ppc['y_obs'], 97.5, axis=0)
            
            return upper - lower
            
        except Exception as e:
            logger.error(f"Uncertainty estimation failed: {e}")
            return np.full(len(X), np.nan)


class NetworkAwareClusterer:
    """Production-ready clustering with network awareness and stability checks."""
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_model = None
        self.stability_score = None
        self.fitted = False
    
    def fit_predict(self, facilities_df: pd.DataFrame) -> np.ndarray:
        """Fit clustering model with stability validation."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            from scipy.spatial.distance import pdist, squareform
            
            # Feature engineering for clustering
            features = self._engineer_clustering_features(facilities_df)
            
            # Multiple distance metrics
            distance_matrices = self._compute_distance_matrices(features, facilities_df)
            
            # Ensemble clustering with stability check
            best_clustering = None
            best_score = -1
            
            for weight_geo, weight_feature in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]:
                combined_distance = (weight_geo * distance_matrices['geographic'] + 
                                   weight_feature * distance_matrices['feature'])
                
                clustering = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    affinity='precomputed',
                    linkage='average'
                )
                
                labels = clustering.fit_predict(combined_distance)
                
                # Evaluate clustering quality
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(features, labels)
                    if score > best_score:
                        best_score = score
                        best_clustering = labels
            
            if best_clustering is None:
                raise RuntimeError("Clustering failed for all configurations")
            
            self.cluster_model = best_clustering
            self.stability_score = best_score
            self.fitted = True
            
            logger.info(f"Clustering completed with silhouette score: {best_score:.3f}")
            return best_clustering
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise RuntimeError(f"Clustering failed: {e}")
    
    def _engineer_clustering_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features for clustering."""
        features = []
        
        # Capacity features
        features.append(df['staff_count'].values)
        features.append(df['cases_detected'].values)
        
        # Access features
        features.append(df['travel_time_hours'].values)
        features.append(df['population_density'].values)
        
        # Technology features
        features.append(df['has_gene_xpert'].astype(int).values)
        features.append(df['has_mobile_xray'].astype(int).values)
        
        return np.column_stack(features)
    
    def _compute_distance_matrices(self, features: np.ndarray, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute multiple distance matrices."""
        from scipy.spatial.distance import pdist, squareform
        
        # Feature-based distance
        feature_dist = squareform(pdist(features, metric='euclidean'))
        
        # Simulate geographic distance (in production, use actual coordinates)
        np.random.seed(self.random_state)
        n_facilities = len(df)
        coords = np.column_stack([
            np.random.uniform(0, 100, n_facilities),
            np.random.uniform(0, 100, n_facilities)
        ])
        geo_dist = squareform(pdist(coords, metric='euclidean'))
        
        return {
            'feature': feature_dist,
            'geographic': geo_dist
        }


# =============================================================================
# OPTIMIZATION ENGINE
# =============================================================================

class ResourceOptimizer:
    """Production-ready multi-objective optimization with constraint handling."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.optimization_history = []
    
    async def optimize_allocation(self, 
                                facilities_df: pd.DataFrame,
                                request: OptimizationRequest) -> AllocationResult:
        """Optimize resource allocation with multiple objectives."""
        
        try:
            # Input validation
            if request.budget <= 0:
                raise ValueError("Budget must be positive")
            
            # Prepare optimization problem
            problem_data = self._prepare_optimization_problem(facilities_df, request)
            
            # Multi-objective optimization
            result = await self._solve_multi_objective(problem_data, request)
            
            # Validate and format result
            allocation_result = self._format_result(result, problem_data, request)
            
            # Store optimization history
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'request': request.dict(),
                'result': allocation_result.dict()
            })
            
            logger.info(f"Optimization completed: {allocation_result.facilities_funded} facilities funded")
            return allocation_result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Optimization failed: {e}")
    
    def _prepare_optimization_problem(self, 
                                    df: pd.DataFrame, 
                                    request: OptimizationRequest) -> Dict:
        """Prepare optimization problem data."""
        
        # Filter facilities by priority regions if specified
        if request.priority_regions:
            df_filtered = df[df['region'].isin(request.priority_regions)]
        else:
            df_filtered = df.copy()
        
        n_facilities = len(df_filtered)
        
        # Intervention costs (simplified model)
        intervention_costs = {
            'gene_xpert': 15000,
            'mobile_xray': 25000,
            'training': 8000
        }
        
        # Expected impacts
        expected_impacts = self._calculate_expected_impacts(df_filtered)
        
        # Equity scores (higher for underserved areas)
        equity_scores = self._calculate_equity_scores(df_filtered)
        
        return {
            'facilities': df_filtered,
            'n_facilities': n_facilities,
            'intervention_costs': intervention_costs,
            'expected_impacts': expected_impacts,
            'equity_scores': equity_scores,
            'facility_ids': df_filtered['facility_id'].tolist()
        }
    
    async def _solve_multi_objective(self, problem_data: Dict, request: OptimizationRequest) -> Dict:
        """Solve multi-objective optimization problem."""
        from scipy.optimize import linprog
        import asyncio
        
        # Decision variables: binary selection for each facility-intervention combination
        n_facilities = problem_data['n_facilities']
        n_interventions = len(request.intervention_types)
        n_vars = n_facilities * n_interventions
        
        # Objective: maximize weighted sum of impact and equity
        c = []
        for i in range(n_facilities):
            for j, intervention in enumerate(request.intervention_types):
                impact = problem_data['expected_impacts'][i]
                equity = problem_data['equity_scores'][i]
                
                # Weighted objective
                obj_value = -(
                    (1 - request.equity_weight) * impact +
                    request.equity_weight * equity
                )
                c.append(obj_value)
        
        # Constraints
        A_ub = []
        b_ub = []
        
        # Budget constraint
        cost_constraint = []
        for i in range(n_facilities):
            for j, intervention in enumerate(request.intervention_types):
                cost = problem_data['intervention_costs'][intervention]
                cost_constraint.append(cost)
        
        A_ub.append(cost_constraint)
        b_ub.append(request.budget)
        
        # Each facility can have at most one intervention
        for i in range(n_facilities):
            constraint = [0] * n_vars
            for j in range(n_interventions):
                constraint[i * n_interventions + j] = 1
            A_ub.append(constraint)
            b_ub.append(1)
        
        # Variable bounds (0-1 for binary, relaxed to continuous)
        bounds = [(0, 1) for _ in range(n_vars)]
        
        # Solve with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                ),
                timeout=request.time_horizon * 60  # Use time_horizon as timeout in minutes
            )
            
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            
            return {
                'success': True,
                'solution': result.x,
                'objective_value': -result.fun,
                'status': result.message
            }
            
        except asyncio.TimeoutError:
            logger.error("Optimization timed out")
            raise RuntimeError("Optimization timed out")
    
    def _calculate_expected_impacts(self, df: pd.DataFrame) -> List[float]:
        """Calculate expected impact for each facility."""
        # Simplified impact model based on facility characteristics
        impacts = []
        for _, facility in df.iterrows():
            base_impact = facility['staff_count'] * 2.5  # Cases per staff member
            
            # Adjustments
            if facility['has_gene_xpert']:
                base_impact *= 1.3
            if facility['has_mobile_xray']:
                base_impact *= 1.2
            
            # Access penalty
            access_penalty = min(facility['travel_time_hours'] / 10, 0.5)
            base_impact *= (1 - access_penalty)
            
            impacts.append(max(base_impact, 1))  # Minimum impact of 1
        
        return impacts
    
    def _calculate_equity_scores(self, df: pd.DataFrame) -> List[float]:
        """Calculate equity scores (higher for more underserved areas)."""
        scores = []
        for _, facility in df.iterrows():
            score = 0.5  # Base score
            
            # Rural bonus
            if facility['urban_rural'] == 'Rural':
                score += 0.3
            
            # Travel time bonus (higher for more remote)
            score += min(facility['travel_time_hours'] / 20, 0.2)
            
            # Technology deficit bonus
            if not facility['has_gene_xpert']:
                score += 0.1
            if not facility['has_mobile_xray']:
                score += 0.1
            
            scores.append(min(score, 1.0))  # Cap at 1.0
        
        return scores
    
    def _format_result(self, 
                      result: Dict, 
                      problem_data: Dict, 
                      request: OptimizationRequest) -> AllocationResult:
        """Format optimization result."""
        
        solution = result['solution']
        n_interventions = len(request.intervention_types)
        
        # Parse solution
        selected_facilities = []
        total_cost = 0
        expected_cases = 0
        
        for i in range(problem_data['n_facilities']):
            for j, intervention in enumerate(request.intervention_types):
                var_idx = i * n_interventions + j
                if solution[var_idx] > 0.5:  # Threshold for binary decision
                    facility_id = problem_data['facility_ids'][i]
                    cost = problem_data['intervention_costs'][intervention]
                    impact = problem_data['expected_impacts'][i]
                    
                    selected_facilities.append({
                        'facility_id': facility_id,
                        'intervention': intervention,
                        'cost': cost,
                        'expected_impact': impact
                    })
                    
                    total_cost += cost
                    expected_cases += impact
        
        # Calculate metrics
        expected_dalys = expected_cases * 0.5  # Simplified: 0.5 DALYs per case
        icer = total_cost / expected_dalys if expected_dalys > 0 else float('inf')
        
        # Equity score
        facility_equity_scores = [problem_data['equity_scores'][i] 
                                for i in range(len(selected_facilities))]
        avg_equity_score = np.mean(facility_equity_scores) if facility_equity_scores else 0
        
        # Confidence interval (simplified)
        ci_width = expected_cases * 0.2  # 20% uncertainty
        confidence_interval = (expected_cases - ci_width, expected_cases + ci_width)
        
        return AllocationResult(
            allocation_id=hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            total_cost=total_cost,
            facilities_funded=len(selected_facilities),
            expected_cases_detected=expected_cases,
            expected_dalys_averted=expected_dalys,
            icer=icer,
            equity_score=avg_equity_score,
            confidence_interval=confidence_interval,
            interventions=selected_facilities
        )


# =============================================================================
# FAIRNESS AND EQUITY AUDITOR
# =============================================================================

class FairnessAuditor:
    """Comprehensive fairness and equity auditing system."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.audit_history = []
    
    def audit_allocation(self, 
                        facilities_df: pd.DataFrame,
                        allocation_result: AllocationResult) -> Dict[str, Any]:
        """Comprehensive fairness audit of resource allocation."""
        
        try:
            audit_results = {}
            
            # Get funded facilities
            funded_facility_ids = [intervention['facility_id'] 
                                 for intervention in allocation_result.interventions]
            
            facilities_df['is_funded'] = facilities_df['facility_id'].isin(funded_facility_ids)
            
            # Demographic parity analysis
            audit_results['demographic_parity'] = self._audit_demographic_parity(facilities_df)
            
            # Geographic equity analysis
            audit_results['geographic_equity'] = self._audit_geographic_equity(facilities_df)
            
            # Outcome equity analysis
            audit_results['outcome_equity'] = self._audit_outcome_equity(facilities_df)
            
            # Access equity analysis
            audit_results['access_equity'] = self._audit_access_equity(facilities_df)
            
            # Overall equity score
            audit_results['overall_equity_score'] = self._calculate_overall_equity_score(audit_results)
            
            # Recommendations
            audit_results['recommendations'] = self._generate_equity_recommendations(audit_results)
            
            # Store audit history
            self.audit_history.append({
                'timestamp': datetime.now(),
                'allocation_id': allocation_result.allocation_id,
                'audit_results': audit_results
            })
            
            logger.info(f"Fairness audit completed: Overall equity score {audit_results['overall_equity_score']:.3f}")
            
            return audit_results
            
        except Exception as e:
            logger.error(f"Fairness audit failed: {e}")
            raise RuntimeError(f"Fairness audit failed: {e}")
    
    def _audit_demographic_parity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit demographic parity across urban/rural divide."""
        
        urban_funded_rate = df[df['urban_rural'] == 'Urban']['is_funded'].mean()
        rural_funded_rate = df[df['urban_rural'] == 'Rural']['is_funded'].mean()
        
        parity_ratio = rural_funded_rate / urban_funded_rate if urban_funded_rate > 0 else 0
        
        return {
            'urban_funding_rate': urban_funded_rate,
            'rural_funding_rate': rural_funded_rate,
            'parity_ratio': parity_ratio,
            'meets_parity_threshold': 0.8 <= parity_ratio <= 1.25,  # 80-125% range
            'disparity_magnitude': abs(1 - parity_ratio)
        }
    
    def _audit_geographic_equity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit geographic equity across regions."""
        
        regional_funding = df.groupby('region')['is_funded'].agg(['count', 'sum', 'mean']).to_dict('index')
        
        funding_rates = [stats['mean'] for stats in regional_funding.values()]
        funding_variance = np.var(funding_rates)
        
        # Identify underserved regions
        mean_funding_rate = np.mean(funding_rates)
        underserved_regions = [
            region for region, stats in regional_funding.items()
            if stats['mean'] < mean_funding_rate * 0.7  # 30% below average
        ]
        
        return {
            'regional_funding_rates': {region: stats['mean'] for region, stats in regional_funding.items()},
            'funding_rate_variance': funding_variance,
            'underserved_regions': underserved_regions,
            'geographic_equity_score': max(0, 1 - funding_variance)  # Lower variance = higher equity
        }
    
    def _audit_outcome_equity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit outcome equity in terms of expected impact per population."""
        
        funded_df = df[df['is_funded'] == True]
        
        if len(funded_df) == 0:
            return {'error': 'No funded facilities to audit'}
        
        # Calculate impact per capita by region
        regional_impact = funded_df.groupby('region').agg({
            'cases_detected': 'sum',
            'population_density': 'sum'
        })
        
        regional_impact['impact_per_capita'] = (
            regional_impact['cases_detected'] / regional_impact['population_density']
        )
        
        impact_variance = regional_impact['impact_per_capita'].var()
        
        return {
            'regional_impact_per_capita': regional_impact['impact_per_capita'].to_dict(),
            'impact_variance': impact_variance,
            'outcome_equity_score': max(0, 1 - impact_variance / regional_impact['impact_per_capita'].mean())
        }
    
    def _audit_access_equity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit access equity based on travel time and infrastructure."""
        
        funded_df = df[df['is_funded'] == True]
        unfunded_df = df[df['is_funded'] == False]
        
        if len(funded_df) == 0 or len(unfunded_df) == 0:
            return {'error': 'Insufficient data for access equity audit'}
        
        # Compare access characteristics
        funded_travel_time = funded_df['travel_time_hours'].mean()
        unfunded_travel_time = unfunded_df['travel_time_hours'].mean()
        
        # Technology access
        funded_tech_access = (funded_df['has_gene_xpert'] | funded_df['has_mobile_xray']).mean()
        unfunded_tech_access = (unfunded_df['has_gene_xpert'] | unfunded_df['has_mobile_xray']).mean()
        
        return {
            'funded_avg_travel_time': funded_travel_time,
            'unfunded_avg_travel_time': unfunded_travel_time,
            'travel_time_equity': unfunded_travel_time / funded_travel_time if funded_travel_time > 0 else 0,
            'funded_tech_access_rate': funded_tech_access,
            'unfunded_tech_access_rate': unfunded_tech_access,
            'prioritizes_remote_areas': unfunded_travel_time > funded_travel_time
        }
    
    def _calculate_overall_equity_score(self, audit_results: Dict[str, Any]) -> float:
        """Calculate overall equity score from individual audits."""
        
        scores = []
        
        # Demographic parity score
        if 'demographic_parity' in audit_results:
            dp_score = 1 - audit_results['demographic_parity']['disparity_magnitude']
            scores.append(dp_score)
        
        # Geographic equity score
        if 'geographic_equity' in audit_results:
            ge_score = audit_results['geographic_equity']['geographic_equity_score']
            scores.append(ge_score)
        
        # Outcome equity score
        if 'outcome_equity' in audit_results and 'outcome_equity_score' in audit_results['outcome_equity']:
            oe_score = audit_results['outcome_equity']['outcome_equity_score']
            scores.append(oe_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_equity_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate actionable equity recommendations."""
        
        recommendations = []
        
        # Demographic parity recommendations
        if 'demographic_parity' in audit_results:
            dp = audit_results['demographic_parity']
            if not dp['meets_parity_threshold']:
                if dp['rural_funding_rate'] < dp['urban_funding_rate']:
                    recommendations.append("Increase funding allocation to rural facilities to improve urban-rural equity")
                else:
                    recommendations.append("Consider rebalancing allocation to ensure urban areas are not underserved")
        
        # Geographic recommendations
        if 'geographic_equity' in audit_results:
            underserved = audit_results['geographic_equity'].get('underserved_regions', [])
            if underserved:
                recommendations.append(f"Prioritize funding in underserved regions: {', '.join(underserved)}")
        
        # Access recommendations
        if 'access_equity' in audit_results:
            ae = audit_results['access_equity']
            if not ae.get('prioritizes_remote_areas', False):
                recommendations.append("Consider prioritizing more remote facilities to improve access equity")
        
        return recommendations


# =============================================================================
# API AND SERVICE LAYER
# =============================================================================

class TBOptimizationService:
    """Main service orchestrator with dependency injection."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Initialize components
        self.cache_manager = CacheManager(config)
        self.data_manager = DataManager(config, self.cache_manager)
        self.model = BayesianHierarchicalModel(config)
        self.clusterer = NetworkAwareClusterer()
        self.optimizer = ResourceOptimizer(config)
        self.auditor = FairnessAuditor(config)
        
        # Service state
        self.is_initialized = False
        self.last_model_training = None
    
    async def initialize(self) -> None:
        """Initialize the service with health checks."""
        try:
            logger.info("Initializing TB Optimization Service...")
            
            # Health check for external dependencies
            self._health_check()
            
            # Load initial data for model training
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Last year of data
            
            # In production, this would load real data
            # For now, we'll create synthetic data
            sample_data = self._create_sample_data()
            
            # Train initial model
            await self._train_models(sample_data)
            
            self.is_initialized = True
            logger.info("Service initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise RuntimeError(f"Service initialization failed: {e}")
    
    def _health_check(self) -> None:
        """Perform health checks on external dependencies."""
        checks = []
        
        # Cache health check
        try:
            self.cache_manager.set("health_check", "ok", ttl=60)
            cache_ok = self.cache_manager.get("health_check") == "ok"
            checks.append(("Cache", cache_ok))
        except Exception as e:
            checks.append(("Cache", False))
            logger.warning(f"Cache health check failed: {e}")
        
        # Report health status
        for component, status in checks:
            logger.info(f"{component} health check: {'PASS' if status else 'FAIL'}")
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration (replace with real data loading)."""
        np.random.seed(42)
        
        regions = ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray']
        facility_types = ['Hospital', 'Health Center', 'Health Post']
        
        data = []
        for i in range(200):  # 200 facilities
            for period in range(12):  # 12 periods
                data.append({
                    'facility_id': i,
                    'period': period,
                    'facility_type': np.random.choice(facility_types),
                    'region': np.random.choice(regions),
                    'urban_rural': np.random.choice(['Urban', 'Rural'], p=[0.3, 0.7]),
                    'staff_count': np.random.poisson(20),
                    'cases_detected': max(0, np.random.poisson(30)),
                    'treatment_success_rate': np.random.beta(8, 2),
                    'population_density': np.random.lognormal(5, 1),
                    'travel_time_hours': np.random.gamma(2, 1),
                    'has_gene_xpert': np.random.binomial(1, 0.3),
                    'has_mobile_xray': np.random.binomial(1, 0.2)
                })
        
        return pd.DataFrame(data)
    
    async def _train_models(self, data: pd.DataFrame) -> None:
        """Train models with the provided data."""
        logger.info("Training models...")
        
        # Prepare features and target
        feature_columns = ['staff_count', 'population_density', 'travel_time_hours', 
                          'has_gene_xpert', 'has_mobile_xray']
        
        # Add region as categorical
        data_encoded = pd.get_dummies(data, columns=['region'])
        feature_columns.extend([col for col in data_encoded.columns if col.startswith('region_')])
        
        X = data_encoded[feature_columns]
        y = data_encoded['cases_detected']
        
        # Train Bayesian model
        self.model.fit(X, y)
        
        # Fit clustering
        self.clusterer.fit_predict(data)
        
        self.last_model_training = datetime.now()
        logger.info("Model training completed")
    
    async def optimize_resources(self, request: OptimizationRequest) -> AllocationResult:
        """Main optimization endpoint."""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            logger.info(f"Processing optimization request: budget=${request.budget:,.0f}")
            
            # Load facility data
            sample_data = self._create_sample_data()  # In production: load real data
            
            # Optimize allocation
            allocation_result = await self.optimizer.optimize_allocation(sample_data, request)
            
            # Audit for fairness
            audit_results = self.auditor.audit_allocation(sample_data, allocation_result)
            
            # Log results
            logger.info(f"Optimization completed: {allocation_result.facilities_funded} facilities funded, "
                       f"ICER=${allocation_result.icer:,.0f}/DALY, equity score={allocation_result.equity_score:.3f}")
            
            return allocation_result
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'service_status': 'healthy' if self.is_initialized else 'initializing',
            'last_model_training': self.last_model_training.isoformat() if self.last_model_training else None,
            'model_convergence': self.model.convergence_diagnostics if self.model.fitted else {},
            'optimization_history_count': len(self.optimizer.optimization_history),
            'audit_history_count': len(self.auditor.audit_history),
            'cache_status': 'redis' if self.cache_manager.use_redis else 'memory'
        }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Global service instance
config = SystemConfig()
service = TBOptimizationService(config)

app = FastAPI(
    title="Ethiopia TB Resource Optimization API",
    description="Production-ready AI-powered resource allocation system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event
# =============================================================================
# COMPLETION: FASTAPI APPLICATION & STREAMLIT DASHBOARD
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Complete the startup event handler"""
    try:
        logger.info("Starting Ethiopia TB Optimization API...")
        await service.initialize()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"API startup failed: {e}")
        raise RuntimeError(f"Failed to start API: {e}")

# API Endpoints
@app.post("/api/optimize", response_model=AllocationResult)
async def optimize_resources(request: OptimizationRequest):
    """Main optimization endpoint"""
    return await service.optimize_resources(request)

@app.get("/api/status")
async def get_system_status():
    """System health and status endpoint"""
    return await service.get_system_status()

@app.get("/api/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/api/validate-data")
async def validate_data_quality():
    """Endpoint for data quality validation"""
    try:
        # In production, this would validate real data sources
        sample_data = service._create_sample_data()
        validation_report = service.data_manager._validate_and_clean_data(sample_data)
        return {"valid": True, "report": validation_report}
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/fairness-audit/{allocation_id}")
async def get_fairness_audit(allocation_id: str):
    """Retrieve fairness audit for a specific allocation"""
    for audit in service.auditor.audit_history:
        if audit['allocation_id'] == allocation_id:
            return audit
    raise HTTPException(status_code=404, detail="Allocation not found")

# Model Management Endpoints
@app.post("/api/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining in background"""
    background_tasks.add_task(service._train_models, service._create_sample_data())
    return {"status": "retraining_started", "message": "Models are being retrained in background"}

@app.get("/api/models/performance")
async def get_model_performance():
    """Get current model performance metrics"""
    if not service.model.fitted:
        raise HTTPException(status_code=404, detail="Models not yet trained")
    
    return {
        "convergence_diagnostics": service.model.convergence_diagnostics,
        "last_training": service.last_model_training.isoformat(),
        "training_samples": len(service._create_sample_data())
    }

if __name__ == "__main__":
    """Production deployment entry point"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # Use default logging
        access_log=True
    )

# =============================================================================
# STREAMLIT DASHBOARD COMPLETION
# =============================================================================

def create_production_dashboard():
    """Complete Streamlit dashboard for policymakers"""
    
    st.set_page_config(
        page_title="Ethiopia TB AI Optimizer",
        page_icon="🇪🇹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🇪🇹 Ethiopia TB Resource Optimization Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("🎛️ Optimization Parameters")
    
    budget = st.sidebar.slider(
        "Available Budget (USD Millions)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5
    ) * 1_000_000
    
    equity_weight = st.sidebar.slider(
        "Equity vs Efficiency Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="Higher values prioritize underserved areas"
    )
    
    priority_regions = st.sidebar.multiselect(
        "Priority Regions",
        options=['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray', 'Addis Ababa'],
        default=[]
    )
    
    time_horizon = st.sidebar.selectbox(
        "Planning Horizon (Years)",
        options=[1, 3, 5, 10],
        index=2
    )
    
    # Main Dashboard Layout
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Optimization", "🗺️ Geographic View", "⚖️ Equity Analysis", "🔧 System Status"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Cost-Effectiveness Frontier")
            # Placeholder for CE frontier plot
            st.plotly_chart(create_ce_frontier_plot(), use_container_width=True)
            
            st.subheader("Recommended Allocation")
            if st.button("🚀 Run Optimization", type="primary"):
                with st.spinner("Optimizing resource allocation..."):
                    try:
                        request = OptimizationRequest(
                            budget=budget,
                            equity_weight=equity_weight,
                            priority_regions=priority_regions,
                            time_horizon=time_horizon
                        )
                        result = asyncio.run(service.optimize_resources(request))
                        display_allocation_result(result)
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
        
        with col2:
            st.subheader("Key Metrics")
            display_key_metrics()
    
    with tab2:
        st.subheader("Geographic Resource Distribution")
        # Placeholder for geographic visualization
        st.plotly_chart(create_geographic_plot(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Regional Allocation")
            st.dataframe(create_regional_allocation_table())
        with col2:
            st.subheader("Facility Clusters")
            st.plotly_chart(create_cluster_visualization(), use_container_width=True)
    
    with tab3:
        st.subheader("Fairness and Equity Analysis")
        display_fairness_analysis()
        
        st.subheader("Vulnerable Population Coverage")
        st.plotly_chart(create_equity_plot(), use_container_width=True)
    
    with tab4:
        display_system_status()

def create_ce_frontier_plot():
    """Create cost-effectiveness frontier plot"""
    # Simplified example - in production would use real data
    fig = go.Figure()
    
    # Example data
    interventions = ['GeneXpert Scale-Up', 'Mobile X-Ray', 'Staff Training']
    costs = [15e6, 12e6, 8e6]
    effects = [45000, 38000, 28000]
    
    for i, (cost, effect) in enumerate(zip(costs, effects)):
        fig.add_trace(go.Scatter(
            x=[cost], y=[effect],
            mode='markers',
            marker=dict(size=15),
            name=interventions[i],
            hovertemplate=f"<b>{interventions[i]}</b><br>Cost: ${cost:,.0f}<br>DALYs: {effect:,.0f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Cost-Effectiveness Frontier",
        xaxis_title="Total Cost (USD)",
        yaxis_title="DALYs Averted",
        showlegend=True
    )
    
    return fig

def display_allocation_result(result: AllocationResult):
    """Display optimization results in a user-friendly format"""
    st.success("🎯 Optimization Completed Successfully!")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${result.total_cost:,.0f}")
    with col2:
        st.metric("Facilities Funded", result.facilities_funded)
    with col3:
        st.metric("DALYs Averted", f"{result.expected_dalys_averted:,.0f}")
    with col4:
        st.metric("ICER", f"${result.icer:,.0f}/DALY")
    
    # Intervention details
    st.subheader("Recommended Interventions")
    interventions_df = pd.DataFrame(result.interventions)
    st.dataframe(interventions_df)
    
    # Download results
    csv = interventions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Allocation Plan",
        data=csv,
        file_name=f"tb_allocation_{result.allocation_id}.csv",
        mime="text/csv"
    )

def display_system_status():
    """Display system health and status"""
    try:
        status = asyncio.run(service.get_system_status())
        
        st.subheader("System Health")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if status['service_status'] == 'healthy' else "🔴"
            st.metric("Service Status", f"{status_color} {status['service_status']}")
        
        with col2:
            st.metric("Model Status", "Trained" if status['last_model_training'] else "Not Trained")
        
        with col3:
            st.metric("Cache", status['cache_status'])
        
        # Model convergence details
        if status['model_convergence']:
            st.subheader("Model Diagnostics")
            conv = status['model_convergence']
            if conv.get('converged', False):
                st.success("✅ Models have converged successfully")
            else:
                st.warning("⚠️ Model convergence requires attention")
            
            st.json(conv)
        
    except Exception as e:
        st.error(f"Failed to retrieve system status: {e}")

# =============================================================================
# PRODUCTION DEPLOYMENT ENHANCEMENTS
# =============================================================================

class ProductionMonitor:
    """Enhanced monitoring for production deployment"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.performance_metrics = []
        self.alert_thresholds = {
            'response_time': 5.0,  # seconds
            'error_rate': 0.05,    # 5%
            'memory_usage': 0.8    # 80%
        }
    
    async def monitor_performance(self):
        """Continuous performance monitoring"""
        while True:
            metrics = {
                'timestamp': datetime.now(),
                'memory_usage': self._get_memory_usage(),
                'active_connections': self._get_active_connections(),
                'response_times': self._get_response_times()
            }
            self.performance_metrics.append(metrics)
            
            # Keep only last 24 hours of metrics
            cutoff = datetime.now() - timedelta(hours=24)
            self.performance_metrics = [
                m for m in self.performance_metrics 
                if m['timestamp'] > cutoff
            ]
            
            # Check alerts
            await self._check_alerts()
            
            await asyncio.sleep(60)  # Check every minute
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        import psutil
        return psutil.virtual_memory().percent / 100
    
    def _get_active_connections(self) -> int:
        """Get number of active database connections"""
        # Implementation would depend on database backend
        return 0
    
    def _get_response_times(self) -> List[float]:
        """Get recent API response times"""
        # Would integrate with API metrics
        return []
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        if len(self.performance_metrics) < 2:
            return
        
        current = self.performance_metrics[-1]
        
        # Memory alert
        if current['memory_usage'] > self.alert_thresholds['memory_usage']:
            logger.warning(f"High memory usage: {current['memory_usage']:.1%}")

# =============================================================================
# FINAL COMPLETION AND INTEGRATION
# =============================================================================

def main():
    """Main entry point for the complete system"""
    print("=" * 70)
    print("🇪🇹 Ethiopia TB AI Optimization System - Production Ready")
    print("=" * 70)
    
    # Initialize configuration
    config = SystemConfig()
    
    # Check environment
    if os.getenv('DEPLOYMENT_ENV') == 'production':
        print("🚀 Starting in PRODUCTION mode")
        # Additional production-specific initialization
        monitor = ProductionMonitor(config)
        asyncio.create_task(monitor.monitor_performance())
    else:
        print("🔧 Starting in DEVELOPMENT mode")
    
    # Start the application
    if len(sys.argv) > 1 and sys.argv[1] == 'dashboard':
        print("📊 Launching Streamlit Dashboard...")
        print("👉 Open http://localhost:8501 in your browser")
        # This would typically be in a separate file
        create_production_dashboard()
    else:
        print("🌐 Starting FastAPI Server...")
        print("👉 API available at http://localhost:8000")
        print("👉 Documentation at http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
