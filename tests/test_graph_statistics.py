from graph.graph_builder import GraphBuilder


def test_graph_statistics_include_topology_and_sector_homophily_denominator():
    builder = object.__new__(GraphBuilder)
    builder.ticker_to_sector = {
        "A": "Technology",
        "B": "Technology",
        "C": "Finance",
        "D": "Unknown",
    }
    builder.graph_mode = "knn_mst"
    builder.graph_window = 10
    builder.max_k = 2
    builder.effective_k = 2
    edges = [
        (0, 1, 0.9),
        (1, 2, 0.8),
        (2, 3, 0.7),
    ]

    stats = builder.compute_graph_stats(["A", "B", "C", "D"], edges)

    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 3
    assert stats["mean_degree"] == 1.5
    assert stats["connected_components"] == 1
    assert stats["isolated_nodes"] == 0
    assert stats["sector_homophilous_edges"] == 1
    assert stats["sector_homophily_eligible_edges"] == 2
    assert stats["sector_edge_homophily"] == 0.5


def test_empty_cross_asset_graph_reports_components_without_fake_homophily():
    builder = object.__new__(GraphBuilder)
    builder.ticker_to_sector = {"A": "Technology", "B": "Finance"}
    builder.graph_mode = "knn_mst"
    builder.graph_window = 10
    builder.max_k = 0
    builder.effective_k = 0

    stats = builder.compute_graph_stats(["A", "B"], [])

    assert stats["num_edges"] == 0
    assert stats["connected_components"] == 2
    assert stats["isolated_nodes"] == 2
    assert stats["sector_homophily_eligible_edges"] == 0
    assert stats["sector_edge_homophily"] != stats["sector_edge_homophily"]
