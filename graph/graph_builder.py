import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GRAPH_MODE_CHOICES = ("knn", "mst", "knn_mst")


def add_graph_args(parser):
    """Register graph-construction CLI arguments on an existing parser.

    graph_window is intentionally not exposed here. Pipeline enforces:
        graph_window == seq_len
    so there is one temporal-window source of truth.
    """
    parser.add_argument("--graph_mode", choices=GRAPH_MODE_CHOICES, default="knn_mst")
    return parser


class GraphBuilder:
    """
    Builds a graph between stocks using scalar statistical features or learned embeddings.

    The graph feature window is supplied by Pipeline and is enforced there to equal seq_len.
    GraphBuilder treats graph_window as a resolved integer, not as an independent config knob.
    """

    def __init__(
        self,
        dfFeats: Dict[str, pd.DataFrame],
        max_k: int,
        n_pca: int = 3,
        ticker_to_sector: Optional[Dict[str, str]] = None,
        graph_embed: str = "pca",
        ablate_feature: str = "none",
        graph_window: int = 10,
        graph_mode: str = "knn_mst",
    ):
        graph_window = int(graph_window)
        if graph_window <= 0:
            raise ValueError("graph_window must be a positive integer.")
        if graph_mode not in GRAPH_MODE_CHOICES:
            raise ValueError(f"graph_mode must be one of {GRAPH_MODE_CHOICES}.")

        self.dfFeats = dfFeats
        self.max_k = int(max_k)
        self.n_pca = int(n_pca)
        self._embeddings: Optional[np.ndarray] = None
        self.ticker_to_sector: Dict[str, str] = ticker_to_sector or {}
        self.graph_embed = graph_embed
        self.ablate_feature = ablate_feature
        self.graph_window = graph_window
        self.graph_mode = graph_mode
        self.effective_k = None
        self.graph_stats: Dict[str, object] = {}
        self.edge_weights: Optional[List[Tuple[int, int, float]]] = None
        self._update_tickers()

    def _rolling_scalar(self, df: pd.DataFrame, column: str) -> float:
        if column not in df.columns:
            return 0.0
        values = pd.to_numeric(df[column], errors="coerce").dropna().tail(self.graph_window)
        if values.empty:
            return 0.0
        return float(values.mean())

    def _compute_scalars(self):
        vectors = []
        valid_tickers = []

        for ticker in self.tickers:
            df = self.dfFeats[ticker]
            try:
                ret = self._rolling_scalar(df, "return")
                vol = self._rolling_scalar(df, "volatility")
                mom = self._rolling_scalar(df, "momentum")

                if self.ablate_feature == "return":
                    ret = 0.0
                elif self.ablate_feature == "volatility":
                    vol = 0.0
                elif self.ablate_feature == "momentum":
                    mom = 0.0

                vectors.append([ret, vol, mom])
                valid_tickers.append(ticker)
            except Exception as exc:
                logging.warning("[GraphBuilder] Error extracting features for %s: %s", ticker, exc)

        if not vectors:
            raise ValueError("[GraphBuilder] No valid scalar features found for graph construction.")

        return valid_tickers, np.array(vectors, dtype=float)

    def _embed_pca(self, mat: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat)

        n_samples, n_features = mat_scaled.shape
        n_comp = min(self.n_pca, n_samples, n_features)

        if self.graph_embed == "raw":
            proj = mat_scaled
            if proj.shape[1] < self.n_pca:
                pad = np.zeros((proj.shape[0], self.n_pca - proj.shape[1]))
                proj = np.hstack([proj, pad])
            logging.info("[GraphBuilder] graph_embed=raw (StandardScaler only), dims=%s", proj.shape)
            return proj

        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(mat_scaled)
        logging.info("[GraphBuilder] graph_embed=pca | PCA variance explained: %s", pca.explained_variance_ratio_)

        if proj.shape[1] < self.n_pca:
            pad = np.zeros((proj.shape[0], self.n_pca - proj.shape[1]))
            proj = np.hstack([proj, pad])

        return proj

    def _similarity_matrix(self, coords: np.ndarray) -> np.ndarray:
        sim = cosine_similarity(coords)
        np.fill_diagonal(sim, -1.0)
        return sim

    def _dedupe_edges(self, edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        edge_map = {}
        for i, j, w in edges:
            if i == j:
                continue
            a, b = sorted((int(i), int(j)))
            edge_map[(a, b)] = max(float(w), edge_map.get((a, b), float("-inf")))
        return [(i, j, w) for (i, j), w in sorted(edge_map.items())]

    def _density(self, n_nodes: int, edges: List[Tuple[int, int, float]]) -> float:
        possible = n_nodes * (n_nodes - 1) / 2
        return (len(self._dedupe_edges(edges)) / possible) if possible > 0 else 0.0

    def _prune_cosine_graph(self, coords: np.ndarray) -> List[Tuple[int, int, float]]:
        sim = self._similarity_matrix(coords)
        n = sim.shape[0]
        k = min(self.max_k, n - 1)
        self.effective_k = k

        if k <= 0:
            return []

        edges = []
        for i in range(n):
            topk = np.argpartition(-sim[i], k)[:k]
            for j in topk:
                if sim[i, j] > 0:
                    a, b = sorted((i, int(j)))
                    edges.append((a, b, float(sim[i, j])))

        edges = self._dedupe_edges(edges)
        logging.info(
            "[GraphBuilder] kNN graph: requested_k=%d effective_k=%d edges=%d density=%.6f",
            self.max_k,
            k,
            len(edges),
            self._density(n, edges),
        )
        return edges

    def _mst_edges(self, coords: np.ndarray) -> List[Tuple[int, int, float]]:
        sim = self._similarity_matrix(coords)
        n = sim.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                similarity = float(sim[i, j])
                graph.add_edge(i, j, weight=1.0 - similarity, similarity=similarity)

        mst = nx.minimum_spanning_tree(graph, weight="weight")
        edges = [(u, v, float(graph[u][v]["similarity"])) for u, v in mst.edges()]
        edges = self._dedupe_edges(edges)

        logging.info("[GraphBuilder] MST graph contains %d edges", len(edges))
        return edges

    def _select_edges_by_mode(self, coords: np.ndarray):
        knn_edges = self._prune_cosine_graph(coords)
        mst_edges = self._mst_edges(coords)

        if self.max_k <= 0:
            return [], mst_edges

        if self.graph_mode == "knn":
            return knn_edges, mst_edges
        if self.graph_mode == "mst":
            return mst_edges, mst_edges
        return self._dedupe_edges(mst_edges + knn_edges), mst_edges

    def compute_graph_stats(self, tickers, edges, mst_edges=None) -> Dict[str, object]:
        n_nodes = len(tickers)
        stats = {
            "num_nodes": n_nodes,
            "num_edges": len(self._dedupe_edges(edges)),
            "density": self._density(n_nodes, edges),
            "homophily": self.sector_homophily_from_edges(tickers, edges),
            "graph_mode": self.graph_mode,
            "graph_window": self.graph_window,
            "requested_k": self.max_k,
            "effective_k": self.effective_k,
        }
        if mst_edges is not None:
            stats["num_mst_edges"] = len(self._dedupe_edges(mst_edges))
        return stats

    def save_graph_stats(self, output_path: str = "results/graph_logging/graph_stats.json") -> None:
        if not self.graph_stats:
            raise RuntimeError("No graph_stats available. Call getLightGraph() before saving stats.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            key: (None if isinstance(value, float) and np.isnan(value) else value)
            for key, value in self.graph_stats.items()
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)
        logging.info("[GraphBuilder] Saved graph stats to %s", path)

    def getLightGraph(self, coords_override: Optional[np.ndarray] = None):
        if coords_override is not None:
            coords = coords_override
            logging.info("[GraphBuilder] Using override coordinates")
        elif self._embeddings is not None:
            coords = self._embeddings
            logging.info("[GraphBuilder] Using learned embeddings")
        else:
            self.tickers, scalar_mat = self._compute_scalars()
            coords = self._embed_pca(scalar_mat)

        selected_edges, mst_edges = self._select_edges_by_mode(coords)
        self.edge_weights = selected_edges
        self.graph_stats = self.compute_graph_stats(self.tickers, selected_edges, mst_edges)

        logging.info(
            "[GraphBuilder] Final graph stats | nodes=%d edges=%d density=%.6f homophily=%s mode=%s window=%d",
            self.graph_stats["num_nodes"],
            self.graph_stats["num_edges"],
            self.graph_stats["density"],
            self.graph_stats["homophily"],
            self.graph_mode,
            self.graph_window,
        )
        return self.tickers, coords, selected_edges, mst_edges

    def buildNetworkX(self, tickers, coords, edges):
        graph = nx.Graph()
        for i, ticker in enumerate(tickers):
            sector = self.ticker_to_sector.get(ticker, "Unknown")
            graph.add_node(ticker, pos=tuple(coords[i]), sector=sector)
        for i, j, w in edges:
            graph.add_edge(tickers[i], tickers[j], weight=round(w, 2))
        return graph

    def sector_homophily_from_edges(self, tickers, edges, ignore_unknown=True, ignore_self_loops=True) -> float:
        if not edges:
            return float("nan")

        undirected = set()
        for i, j, _ in edges:
            a, b = (i, j) if i <= j else (j, i)
            undirected.add((a, b))

        same = 0
        total = 0
        for i, j in undirected:
            if ignore_self_loops and i == j:
                continue
            si = self.ticker_to_sector.get(tickers[i], "Unknown")
            sj = self.ticker_to_sector.get(tickers[j], "Unknown")
            if ignore_unknown and ("Unknown" in (si, sj) or si is None or sj is None):
                continue
            total += 1
            if si == sj:
                same += 1
        return (same / total) if total > 0 else float("nan")

    def sector_homophily_from_edge_index(self, tickers, edge_index: torch.Tensor, ignore_unknown=True, ignore_self_loops=True) -> float:
        if edge_index is None or edge_index.numel() == 0:
            return float("nan")

        edges = set()
        for src, dst in edge_index.detach().cpu().t().tolist():
            a, b = (src, dst) if src <= dst else (dst, src)
            edges.add((a, b))

        return self.sector_homophily_from_edges(tickers, [(i, j, 1.0) for i, j in edges], ignore_unknown, ignore_self_loops)

    def build_edge_weight_tensor(self, edge_index: torch.Tensor) -> torch.Tensor:
        if self.edge_weights is None:
            raise AttributeError("GraphBuilder has no edge_weights attribute. Call getLightGraph() first.")

        edge_weight_map = {(min(i, j), max(i, j)): w for i, j, w in self.edge_weights}
        weights = []
        for src, dst in edge_index.T.tolist():
            key = (min(src, dst), max(src, dst))
            weights.append(edge_weight_map.get(key, 0.0))
        return torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)

    def updateFeatures(self, newFeats: Dict[str, pd.DataFrame]) -> None:
        self.dfFeats = newFeats
        self._update_tickers()
        logging.info("[GraphBuilder] Updated features for %d tickers", len(self.tickers))

    def set_node_embeddings(self, embeddings: np.ndarray) -> None:
        if embeddings.shape[0] != len(self.tickers):
            raise ValueError("Embeddings must have one row per ticker.")
        self._embeddings = embeddings
        logging.info("[GraphBuilder] Stored learned node embeddings of shape %s", embeddings.shape)

    def get_node_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            raise RuntimeError("No learned embeddings have been set.")
        return self._embeddings

    def _update_tickers(self) -> None:
        self.tickers = list(self.dfFeats.keys())

    def get_max_k(self):
        return self.max_k

    def set_sector_map(self, ticker_to_sector: Dict[str, str]) -> None:
        self.ticker_to_sector = ticker_to_sector or {}
        logging.info("[GraphBuilder] Sector map set for %d tickers", len(self.ticker_to_sector))
