import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class GraphBuilder:
    """
    Builds a graph between stocks using scalar statistical features or learned embeddings.
    Outputs a lightweight, pruned similarity graph for downstream STGNN processing.
    """
    def __init__(self, dfFeats: Dict[str, pd.DataFrame], max_k: int, n_pca: int = 3):
        # === STEP 1: Initialise Configuration ===
        # ------------------------------------
        self.dfFeats = dfFeats
        self.max_k = max_k
        self.n_pca = n_pca
        self._embeddings: Optional[np.ndarray] = None
        self._update_tickers()
    
    def _compute_scalars(self):
        # === STEP 1: Clear Current Scalars ===
        # ------------------------------------
        vectors = []
        valid_tickers = []

		# === STEP 2: Build Vectors ===
        # ------------------------------------
        for t in self.tickers:
            #   1. Grab scalars (features) per stock
            df = self.dfFeats[t]
            try:
                #   2. Store individual scalars
                ret = df['return'].iloc[-1]
                vol = df['volatility'].iloc[-1]
                mom = df.get('momentum', pd.Series([0], index=[df.index[-1]])).iloc[-1]

                #   3. Create and store vector from scalars
                vectors.append([ret, vol, mom])
                valid_tickers.append(t)

            except Exception as e:
                logging.warning("[GraphBuilder] Error extracting features for %s: %s", t, e)

        if not vectors:
            raise ValueError("[GraphBuilder] No valid scalar features found for graph construction.")

        #   4.  Pass all vectors for graph building
        mat = np.array(vectors)
        return valid_tickers, mat

    def _embed_pca(self, mat: np.ndarray) -> np.ndarray:
		# === STEP 1: Normalisation ===
        # ------------------------------------
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat)

        # === STEP 2: Component Counting ===
        # ------------------------------------
        n_samples, n_features = mat_scaled.shape
        n_comp = min(self.n_pca, n_samples, n_features)

        # === STEP 3: PCA Projection ===
        # ------------------------------------

        #   1. Establish dimension size (should be 3)
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(mat_scaled)
        logging.info("[GraphBuilder] PCA variance explained: %s", pca.explained_variance_ratio_)

        #   2. Force 3 dimensions if less than 3 scalars passed
        if proj.shape[1] < self.n_pca:
            pad = np.zeros((proj.shape[0], self.n_pca - proj.shape[1]))
            proj = np.hstack([proj, pad])
        return proj

    def _prune_cosine_graph(self, coords: np.ndarray) -> List[Tuple[int, int, float]]:
        sim = cosine_similarity(coords)
        np.fill_diagonal(sim, -1)  # prevent self-loops

        # Effective k actually used
        k = min(self.max_k, sim.shape[0] - 1)
        self.effective_k = k  # <--- add this so we can log it outside

        edge_set = set()

        for i in range(sim.shape[0]):
            # IMPORTANT: use k, not self.max_k
            topk = np.argpartition(-sim[i], k)[:k]
            for j in topk:
                if sim[i, j] > 0:
                    a, b = sorted((i, j))
                    edge_set.add((a, b, float(sim[i, j])))

        edges = list(edge_set)

        # Extra graph stats for defensibility
        n = sim.shape[0]
        possible_undirected = n * (n - 1) / 2
        density = (len(edges) / possible_undirected) if possible_undirected > 0 else 0.0

        logging.info(
            "[GraphBuilder] Pruned graph: requested_k=%d effective_k=%d edges=%d density=%.6f",
            self.max_k, k, len(edges), density
        )

        return edges

    # Prune -> MST
    def getLightGraph(
        self,
        coords_override: Optional[np.ndarray] = None
    ) -> Tuple[List[str], np.ndarray, List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
		# === STEP 1: Check For Passed Coordinates (For Graph Rebuilding) ===
        # ------------------------------------
        if coords_override is not None:
            coords = coords_override
            logging.info("[GraphBuilder] Using override coordinates")
        elif self._embeddings is not None:
            coords = self._embeddings
            logging.info("[GraphBuilder] Using learned embeddings")
        else:
            self.tickers, scalar_mat = self._compute_scalars()
            coords = self._embed_pca(scalar_mat)

        # === STEP 2: Prune Edges Via Max-k ===
        # ------------------------------------
        pruned = self._prune_cosine_graph(coords)
        self.edge_weights = pruned  # Store for later use in edge weight tensor

        # === STEP 3: Build Graph From Nodes And Edges ===
        # ------------------------------------

        #   1. Define graph
        G = nx.Graph()

        #   2. Add nodes
        for i in range(len(self.tickers)):
            G.add_node(i)

        #   3. Add edges
        for i, j, w in pruned:
            distance = 1 - w  # Convert similarity to distance
            G.add_edge(i, j, weight=distance)

        #   4. Add MST edges (Mantegna et al. 1999)
        mst = nx.minimum_spanning_tree(G, weight='weight')
        mst_edges = [(u, v, 1 - G[u][v]['weight']) for u, v in mst.edges()]
        logging.info("[GraphBuilder] MST contains %d edges", len(mst_edges))

        #   5. Return graph nodes (stocks), coords, max-k edges, MST edges
        return self.tickers, coords, pruned, mst_edges

    """
    # MST -> Prune
    def getLightGraph(
        self,
        coords_override: Optional[np.ndarray] = None
    ) -> Tuple[List[str], np.ndarray, List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
        """
        #   Returns:
        #   - tickers
        #   - coords (Nx3): node embeddings or PCA-projected scalars
        #   - all_edges: MST + top-k augmentations (non-duplicate)
        #   - mst_edges: pure minimum spanning tree edges
    """
        if coords_override is not None:
            coords = coords_override
            logging.info("[GraphBuilder] Using override coordinates")
        elif self._embeddings is not None:
            coords = self._embeddings
            logging.info("[GraphBuilder] Using learned embeddings")
        else:
            self.tickers, scalar_mat = self._compute_scalars()
            coords = self._embed_pca(scalar_mat)

        sim = cosine_similarity(coords)
        np.fill_diagonal(sim, -1)

        # Build full graph for MST
        full_G = nx.Graph()
        for i in range(len(self.tickers)):
            full_G.add_node(i)
        for i in range(sim.shape[0]):
            for j in range(i + 1, sim.shape[1]):
                if sim[i, j] > 0:
                    full_G.add_edge(i, j, weight=1 - sim[i, j])

        mst = nx.minimum_spanning_tree(full_G, weight='weight')
        mst_edges = [(u, v, 1 - full_G[u][v]['weight']) for u, v in mst.edges()]

        logging.info("[GraphBuilder] MST contains %d edges", len(mst_edges))

        # Now: add top-k edges per node from similarity matrix
        k = min(self.max_k, sim.shape[0] - 1)
        extra_edges = set()

        for i in range(sim.shape[0]):
            topk = np.argpartition(-sim[i], k)[:k]
            for j in topk:
                if i == j or sim[i, j] <= 0:
                    continue
                a, b = sorted((i, j))
                extra_edges.add((a, b, float(sim[i, j])))

        # Avoid duplicates with MST
        mst_set = {(min(u, v), max(u, v)) for u, v, _ in mst_edges}
        augmented = [e for e in extra_edges if (e[0], e[1]) not in mst_set]

        combined_edges = mst_edges + augmented
        self.edge_weights = combined_edges  # used for torch edge_index later

        logging.info("[GraphBuilder] Augmented with %d extra top-k edges", len(augmented))
        logging.info("[GraphBuilder] Total edges after MST+top-k: %d", len(combined_edges))

        return self.tickers, coords, combined_edges, mst_edges
        """
    
    def buildNetworkX(self, tickers: List[str], coords: np.ndarray, edges: List[Tuple[int, int, float]]) -> nx.Graph:
        # ====================================
		# === Helper to build graph
        G = nx.Graph()
        for i, t in enumerate(tickers):
            G.add_node(t, pos=tuple(coords[i]))
        for i, j, w in edges:
            G.add_edge(tickers[i], tickers[j], weight=round(w, 2))
        return G
    
    def build_edge_weight_tensor(self, edge_index: torch.Tensor) -> torch.Tensor:
        # ====================================
		# === Helper to build tensor from edge weights
        if not hasattr(self, 'edge_weights') or self.edge_weights is None:
            raise AttributeError("GraphBuilder has no edge_weights attribute. Ensure you store edge weights during pruning.")

        #   1. Create edge weight map
        edge_weight_map = {(i, j): w for i, j, w in self.edge_weights}

        #   2. Extract weights from edge_index
        weights = []
        for src, dst in edge_index.T.tolist():
            key = (min(src, dst), max(src, dst))
            weight = edge_weight_map.get(key, 0.0)
            weights.append(weight)

        #   3. Unsqueeze to add feature dimension
        return torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
    
    def updateFeatures(self, newFeats: Dict[str, pd.DataFrame]) -> None:
        # ====================================
		# === Helper to update graph features
        self.dfFeats = newFeats
        self._update_tickers()
        logging.info("[GraphBuilder] Updated features for %d tickers", len(self.tickers))

    def set_node_embeddings(self, embeddings: np.ndarray) -> None:
        # ====================================
		# === Helper to set node embeddings
        if embeddings.shape[0] != len(self.tickers):
            raise ValueError("Embeddings must have one row per ticker.")
        self._embeddings = embeddings
        logging.info("[GraphBuilder] Stored learned node embeddings of shape %s", embeddings.shape)

    def get_node_embeddings(self) -> np.ndarray:
        # ====================================
		# === Helper to get node embeddings
        if self._embeddings is None:
            raise RuntimeError("No learned embeddings have been set.")
        return self._embeddings

    def _update_tickers(self) -> None:
        # ====================================
		# === Helper to update tickers list
        self.tickers = list(self.dfFeats.keys())

    def get_max_k(self):
        # ====================================
		# === Helper to get max-k (for graph efficiency)
        return self.max_k