"""Clustering utilities for skeleton processor."""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist, cosine
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class SphericalKMeans:
    """Spherical K-means clustering for high-dimensional embeddings."""
    
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, 
                 random_state: int = 42):
        """Initialize Spherical K-means clustering.
        
        Args:
            n_clusters: Number of clusters (K)
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Fitted attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
    def fit(self, X: np.ndarray) -> 'SphericalKMeans':
        """Fit Spherical K-means to data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        logger.info(f"Starting Spherical K-means clustering with K={self.n_clusters}")
        
        # Validate input data
        if X.size == 0:
            raise ValueError("Input array is empty")
        if X.shape[1] == 0:
            raise ValueError("Input array has 0 features")
        if X.shape[0] == 0:
            raise ValueError("Input array has 0 samples")
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"Number of samples ({X.shape[0]}) must be >= n_clusters ({self.n_clusters})")
        
        # Normalize input data to unit sphere
        X_normalized = normalize(X, norm='l2', axis=1)
        
        # Initialize centroids
        np.random.seed(self.random_state)
        self.cluster_centers_ = self._init_centroids(X_normalized)
        
        prev_inertia = float('inf')
        
        for iteration in range(self.max_iter):
            # Assign points to clusters based on cosine similarity
            distances = self._cosine_distances(X_normalized, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centers = self._update_centroids(X_normalized, self.labels_)
            
            # Calculate inertia (sum of cosine distances to centroids)
            self.inertia_ = self._calculate_inertia(X_normalized, self.cluster_centers_, self.labels_)
            
            # Check for convergence
            if abs(prev_inertia - self.inertia_) < self.tol:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            self.cluster_centers_ = new_centers
            prev_inertia = self.inertia_
            
            if iteration % 50 == 0:
                logger.debug(f"Iteration {iteration}, inertia: {self.inertia_:.4f}")
        else:
            logger.warning(f"Did not converge after {self.max_iter} iterations")
        
        self.n_iter_ = iteration + 1
        
        logger.info(f"Clustering completed. Final inertia: {self.inertia_:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_normalized = normalize(X, norm='l2', axis=1)
        distances = self._cosine_distances(X_normalized, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and predict cluster labels.
        
        Args:
            X: Data matrix
            
        Returns:
            Cluster labels
        """
        return self.fit(X).labels_
    
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++ strategy.
        
        Args:
            X: Normalized data matrix
            
        Returns:
            Initial centroids
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Choose remaining centroids using k-means++ strategy
        for c_idx in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = np.array([
                min([1 - np.dot(x, centroid) for centroid in centroids[:c_idx]])
                for x in X
            ])
            
            # Choose next centroid with probability proportional to squared distance
            probs = distances / distances.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[c_idx] = X[j]
                    break
        
        return normalize(centroids, norm='l2', axis=1)
    
    def _cosine_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Calculate cosine distances between points and centroids.
        
        Args:
            X: Data points
            centroids: Cluster centroids
            
        Returns:
            Distance matrix
        """
        # Cosine distance = 1 - cosine_similarity
        similarities = np.dot(X, centroids.T)
        return 1 - similarities
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update cluster centroids.
        
        Args:
            X: Data points
            labels: Current cluster assignments
            
        Returns:
            Updated centroids
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                # Centroid is the mean of assigned points, normalized
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[k] = X[np.random.randint(len(X))]
        
        return normalize(new_centroids, norm='l2', axis=1)
    
    def _calculate_inertia(self, X: np.ndarray, centroids: np.ndarray, 
                          labels: np.ndarray) -> float:
        """Calculate within-cluster sum of cosine distances.
        
        Args:
            X: Data points
            centroids: Cluster centroids
            labels: Cluster assignments
            
        Returns:
            Inertia value
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                distances = 1 - np.dot(cluster_points, centroids[k])
                inertia += distances.sum()
        return inertia


class ClusterAnalyzer:
    """Utility class for analyzing clusters and calculating homogeneity scores."""
    
    @staticmethod
    def calculate_homogeneity_score(embeddings: np.ndarray, labels: np.ndarray, 
                                  cluster_id: int) -> float:
        """Calculate homogeneity score for a cluster.
        
        Args:
            embeddings: All embeddings
            labels: Cluster labels  
            cluster_id: ID of cluster to analyze
            
        Returns:
            Homogeneity score (higher = more homogeneous)
        """
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_embeddings) < 2:
            return 1.0  # Single point clusters are perfectly homogeneous
        
        # Calculate pairwise cosine similarities within cluster
        similarities = []
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                sim = 1 - cosine(cluster_embeddings[i], cluster_embeddings[j])
                similarities.append(sim)
        
        # Return mean similarity as homogeneity score
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def find_medoid(embeddings: np.ndarray, labels: np.ndarray, 
                   cluster_id: int) -> int:
        """Find medoid (most central point) of a cluster.
        
        Args:
            embeddings: All embeddings
            labels: Cluster labels
            cluster_id: ID of cluster
            
        Returns:
            Index of medoid in the cluster
        """
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_embeddings) == 0:
            return -1
        if len(cluster_embeddings) == 1:
            return cluster_indices[0]
        
        # Calculate sum of cosine distances to all other points in cluster
        min_avg_distance = float('inf')
        medoid_idx = -1
        
        for i, embedding in enumerate(cluster_embeddings):
            avg_distance = np.mean([
                cosine(embedding, other) 
                for j, other in enumerate(cluster_embeddings) 
                if i != j
            ])
            
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                medoid_idx = cluster_indices[i]
        
        return medoid_idx
    
    @staticmethod
    def find_farthest_members(embeddings: np.ndarray, labels: np.ndarray,
                            cluster_id: int, medoid_idx: int, n_members: int = 2) -> List[int]:
        """Find farthest members from cluster centroid.
        
        Args:
            embeddings: All embeddings
            labels: Cluster labels
            cluster_id: ID of cluster
            medoid_idx: Index of medoid
            n_members: Number of farthest members to return
            
        Returns:
            List of indices of farthest members
        """
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_embeddings) <= n_members:
            # Return all non-medoid members
            return [idx for idx in cluster_indices if idx != medoid_idx]
        
        # Calculate centroid
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Calculate distances to centroid
        distances = []
        for i, idx in enumerate(cluster_indices):
            if idx != medoid_idx:  # Exclude medoid
                distance = cosine(embeddings[idx], centroid)
                distances.append((distance, idx))
        
        # Sort by distance (descending) and return n_members farthest
        distances.sort(reverse=True)
        return [idx for _, idx in distances[:n_members]]
    
    @staticmethod
    def calculate_optimal_k(embeddings: np.ndarray, max_k: Optional[int] = None,
                          min_k: int = 2) -> int:
        """Calculate optimal number of clusters using elbow method.
        
        Args:
            embeddings: Embedding matrix
            max_k: Maximum K to try (defaults to average paragraphs per document)
            min_k: Minimum K to try
            
        Returns:
            Optimal number of clusters
        """
        if max_k is None:
            max_k = min(50, len(embeddings) // 4)  # Heuristic: quarter of data points
        
        max_k = min(max_k, len(embeddings) - 1)
        
        if max_k < min_k:
            return min_k
        
        inertias = []
        silhouette_scores = []
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = SphericalKMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score for validation
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(embeddings, labels, metric='cosine')
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate K={k}: {e}")
                inertias.append(float('inf'))
                silhouette_scores.append(0)
        
        # Use elbow method: find point with maximum second derivative
        if len(inertias) < 3:
            return min_k
        
        # Calculate second derivatives
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        # Find elbow point
        elbow_idx = np.argmax(second_derivatives)
        optimal_k = elbow_idx + min_k + 1
        
        logger.info(f"Optimal K determined: {optimal_k} (from range {min_k}-{max_k})")
        return optimal_k
