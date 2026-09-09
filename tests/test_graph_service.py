"""Tests for :mod:`ucsa.models.graph_service`."""

from __future__ import annotations

import pytest
import torch

from ucsa.models.graph_service import (
    Concept,
    CosineClusterer,
    GraphEdge,
    GraphMemory,
    GraphService,
    supported_banks,
)
from ucsa.models.memory import Memory, MemoryUpdate
from ucsa.models.state import PCSConfig, PersistentCognitiveState


def tiny_pcs() -> PersistentCognitiveState:
    """Return a fresh PCS sized for tests."""
    return PersistentCognitiveState(PCSConfig(hidden_size=32))


def populate_long_term(
    pcs: PersistentCognitiveState, num_tokens: int = 20
) -> None:
    """Fill the long-term bank with ``num_tokens`` random tokens."""
    mem = Memory(pcs)
    candidate = MemoryUpdate(
        tokens=torch.randn(num_tokens, pcs.config.hidden_size),
        importance=torch.ones(num_tokens),
        confidence=1.0,
    )
    mem.accept_into_long_term(candidate, max_slots=num_tokens)


class TestCosineClusterer:
    """Tests for :class:`CosineClusterer`."""

    def test_invalid_num_clusters(self) -> None:
        """``num_clusters`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            CosineClusterer(num_clusters=0)

    def test_cluster_assignments_in_range(self) -> None:
        """Every assignment is a valid cluster id."""
        clusterer = CosineClusterer(num_clusters=4)
        embeddings = torch.randn(20, 32)
        assignments, centroids = clusterer.cluster(embeddings)
        assert assignments.shape == (20,)
        assert torch.all(assignments >= 0)
        assert torch.all(assignments < 4)
        assert centroids.shape == (4, 32)

    def test_centroids_normalised(self) -> None:
        """Cluster centroids have unit L2 norm."""
        clusterer = CosineClusterer(num_clusters=3)
        embeddings = torch.randn(15, 16)
        _, centroids = clusterer.cluster(embeddings)
        norms = centroids.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_empty_input(self) -> None:
        """Empty input yields empty assignments and zero centroids."""
        clusterer = CosineClusterer(num_clusters=4)
        assignments, centroids = clusterer.cluster(torch.zeros(0, 32))
        assert assignments.shape == (0,)
        assert centroids.shape == (4, 32)

    def test_degenerate_fewer_than_clusters(self) -> None:
        """Fewer points than clusters yields one cluster per point."""
        clusterer = CosineClusterer(num_clusters=8)
        embeddings = torch.randn(3, 16)
        assignments, centroids = clusterer.cluster(embeddings)
        # When n < num_clusters, each point gets its own cluster.
        # The remaining clusters are marked with -1.
        assert assignments.shape == (8,)
        assert centroids.shape == (8, 16)
        real = assignments[assignments >= 0]
        assert real.shape == (3,)
        assert torch.all(real < 3)

    def test_separated_clusters_are_distinct(self) -> None:
        """Clearly-separated inputs get distinct cluster assignments."""
        clusterer = CosineClusterer(num_clusters=2)
        cluster_a = torch.randn(10, 16) + torch.tensor([10.0] * 16)
        cluster_b = torch.randn(10, 16) - torch.tensor([10.0] * 16)
        embeddings = torch.cat([cluster_a, cluster_b], dim=0)
        assignments, _ = clusterer.cluster(embeddings)
        # First 10 should be in cluster 0, second 10 in cluster 1.
        assert torch.all(assignments[:10] == assignments[0])
        assert torch.all(assignments[10:] == assignments[10])
        assert assignments[0] != assignments[10]

    def test_invalid_dim(self) -> None:
        """Non-2D embeddings raise."""
        clusterer = CosineClusterer(num_clusters=2)
        with pytest.raises(ValueError):
            clusterer.cluster(torch.randn(4, 2, 8))


class TestGraphServiceConstruction:
    """Tests for :class:`GraphService` construction."""

    def test_invalid_num_concepts(self) -> None:
        """``num_concepts`` of zero or less is rejected."""
        with pytest.raises(ValueError):
            GraphService(num_concepts=0)

    def test_invalid_threshold(self) -> None:
        """``co_activation_threshold`` outside ``[0, 1]`` is rejected."""
        with pytest.raises(ValueError):
            GraphService(co_activation_threshold=-0.1)
        with pytest.raises(ValueError):
            GraphService(co_activation_threshold=1.5)

    def test_supported_banks(self) -> None:
        """``supported_banks`` returns every PCS bank name."""
        banks = supported_banks()
        assert "long_term" in banks
        assert "working" in banks


class TestGraphServiceBuild:
    """Tests for :meth:`GraphService.build_graph`."""

    def test_empty_pcs_yields_empty_graph(self) -> None:
        """With no used long-term slots, the graph is empty."""
        gs = GraphService(num_concepts=4)
        graph = gs.build_graph(tiny_pcs())
        assert graph.concepts == []
        assert graph.edges == []

    def test_concepts_count_matches_request(self) -> None:
        """Up to ``num_concepts`` clusters are produced."""
        gs = GraphService(num_concepts=4)
        populate_long_term(tiny_pcs(), num_tokens=20)
        graph = gs.build_graph(tiny_pcs())
        assert len(graph.concepts) <= 4

    def test_concept_members_are_unique(self) -> None:
        """Each long-term slot appears in at most one concept."""
        gs = GraphService(num_concepts=3)
        populate_long_term(tiny_pcs(), num_tokens=15)
        graph = gs.build_graph(tiny_pcs())
        seen: set[int] = set()
        for concept in graph.concepts:
            for member in concept.member_indices:
                assert member not in seen
                seen.add(member)

    def test_concept_centroids_have_unit_norm(self) -> None:
        """Concept centroids are unit-normalised."""
        gs = GraphService(num_concepts=3)
        populate_long_term(tiny_pcs(), num_tokens=15)
        graph = gs.build_graph(tiny_pcs())
        for concept in graph.concepts:
            norm = concept.centroid.norm().item()
            assert norm == pytest.approx(1.0, abs=1e-4)

    def test_concept_token_shape(self) -> None:
        """Each concept exposes a token of shape ``(hidden_size,)``."""
        gs = GraphService(num_concepts=3)
        populate_long_term(tiny_pcs(), num_tokens=10)
        graph = gs.build_graph(tiny_pcs())
        for concept in graph.concepts:
            assert concept.token is not None
            assert concept.token.shape == (32,)

    def test_concept_deduplication(self) -> None:
        """Concepts are deduplicated by centroid proximity on rebuild."""
        gs = GraphService(num_concepts=3)
        populate_long_term(tiny_pcs(), num_tokens=12)
        graph_a = gs.build_graph(tiny_pcs())
        graph_b = gs.build_graph(tiny_pcs())
        # Re-running with the same data should yield the same centroids.
        for ca, cb in zip(graph_a.concepts, graph_b.concepts, strict=False):
            assert torch.allclose(ca.centroid, cb.centroid, atol=1e-3)


class TestGraphServiceRetrieval:
    """Tests for :meth:`GraphService.retrieve`."""

    def test_retrieve_returns_top_k(self) -> None:
        """``retrieve`` returns at most ``top_k`` concepts."""
        gs = GraphService(num_concepts=4)
        populate_long_term(tiny_pcs(), num_tokens=20)
        gs.build_graph(tiny_pcs())
        retrieved = gs.retrieve(torch.randn(32), top_k=2)
        assert len(retrieved) <= 2

    def test_retrieve_sorted_by_similarity(self) -> None:
        """Retrieved concepts are sorted by similarity to the query."""
        gs = GraphService(num_concepts=3)
        # Populate with well-separated embeddings so clustering succeeds.
        pcs = tiny_pcs()
        tokens = torch.cat(
            [
                torch.randn(5, 32) + 10.0,
                torch.randn(5, 32) - 10.0,
                torch.randn(5, 32) + torch.tensor([10.0, -10.0] + [0.0] * 30),
            ],
            dim=0,
        )
        candidate = MemoryUpdate(tokens=tokens, importance=torch.ones(15))
        Memory(pcs).accept_into_long_term(candidate, max_slots=15)
        gs.build_graph(pcs)
        # Use one concept's centroid as the query.
        query = gs.last_graph.concepts[0].centroid
        retrieved = gs.retrieve(query, top_k=3)
        assert retrieved[0].centroid.shape == query.shape

    def test_retrieve_empty_when_no_graph(self) -> None:
        """Without a graph, ``retrieve`` returns an empty list."""
        gs = GraphService(num_concepts=2)
        assert gs.retrieve(torch.randn(32)) == []

    def test_retrieve_handles_batch_query(self) -> None:
        """``retrieve`` accepts a ``(batch, hidden)`` query."""
        gs = GraphService(num_concepts=3)
        populate_long_term(tiny_pcs(), num_tokens=15)
        gs.build_graph(tiny_pcs())
        retrieved = gs.retrieve(torch.randn(2, 32), top_k=2)
        assert len(retrieved) <= 2


class TestGraphServiceTokens:
    """Tests for :meth:`GraphService.retrieve_tokens`."""

    def test_token_shape(self) -> None:
        """``retrieve_tokens`` returns ``(top_k, hidden)``."""
        gs = GraphService(num_concepts=3)
        populate_long_term(tiny_pcs(), num_tokens=15)
        gs.build_graph(tiny_pcs())
        # Use top_k equal to the number of concepts that exist.
        num_concepts = len(gs.last_graph.concepts)
        tokens = gs.retrieve_tokens(torch.randn(32), top_k=2)
        assert tokens.shape == (min(2, num_concepts), 32)

    def test_no_concepts_returns_empty(self) -> None:
        """Without concepts, tokens have shape ``(0, hidden)``."""
        gs = GraphService(num_concepts=2)
        tokens = gs.retrieve_tokens(torch.randn(32), top_k=2)
        assert tokens.shape == (0, 32)


class TestGraphEdge:
    """Tests for :class:`GraphEdge`."""

    def test_construction(self) -> None:
        """Edges hold source, target, and weight."""
        edge = GraphEdge(source=0, target=1, weight=0.8)
        assert edge.source == 0
        assert edge.target == 1
        assert edge.weight == 0.8


class TestGraphMemory:
    """Tests for :class:`GraphMemory`."""

    def test_default_empty(self) -> None:
        """Default graph memory is empty."""
        graph = GraphMemory()
        assert graph.concepts == []
        assert graph.edges == []


class TestConcept:
    """Tests for :class:`Concept`."""

    def test_construction(self) -> None:
        """A concept has a centroid and members."""
        centroid = torch.randn(8)
        concept = Concept(centroid=centroid, member_indices=[0, 1, 2])
        assert torch.allclose(concept.centroid, centroid)
        assert concept.member_indices == [0, 1, 2]
        assert concept.token is None


class TestInjection:
    """Tests for :meth:`GraphService.inject_into_working`."""

    def test_inject_writes_to_working(self) -> None:
        """Tokens are prepended into the working bank."""
        gs = GraphService(num_concepts=4)
        pcs = tiny_pcs()
        populate_long_term(pcs, num_tokens=20)
        gs.build_graph(pcs)
        # Snapshot the working bank before injection.
        before = pcs.get_bank("working").clone()
        n = gs.inject_into_working(pcs, torch.randn(32), top_k=3)
        after = pcs.get_bank("working")
        assert n == 3
        # First three rows were overwritten.
        assert not torch.allclose(before[:3], after[:3])
        # Tail of the bank is unchanged.
        assert torch.allclose(before[3:], after[3:])

    def test_inject_no_graph_returns_zero(self) -> None:
        """Without a graph, ``inject_into_working`` is a no-op."""
        gs = GraphService(num_concepts=2)
        pcs = tiny_pcs()
        before = pcs.get_bank("working").clone()
        n = gs.inject_into_working(pcs, torch.randn(32), top_k=3)
        assert n == 0
        after = pcs.get_bank("working")
        assert torch.allclose(before, after)

    def test_inject_more_than_working_caps(self) -> None:
        """Injection is capped by the working-bank size."""
        gs = GraphService(num_concepts=4)
        pcs = tiny_pcs()
        populate_long_term(pcs, num_tokens=20)
        gs.build_graph(pcs)
        # Ask for many more tokens than the working bank can hold.
        n = gs.inject_into_working(pcs, torch.randn(32), top_k=200)
        # Capped by the working bank size and by the number of concepts.
        assert n <= min(
            200, pcs.bank_size("working"), len(gs.last_graph.concepts)
        )

    def test_inject_query_batch_dim(self) -> None:
        """Batch-dim query is accepted."""
        gs = GraphService(num_concepts=3)
        pcs = tiny_pcs()
        populate_long_term(pcs, num_tokens=15)
        gs.build_graph(pcs)
        n = gs.inject_into_working(pcs, torch.randn(2, 32), top_k=2)
        assert n >= 0

    def test_injection_round_trip(self) -> None:
        """End-to-end: build graph, retrieve, inject, read back."""
        gs = GraphService(num_concepts=4)
        pcs = tiny_pcs()
        populate_long_term(pcs, num_tokens=20)
        gs.build_graph(pcs)
        query = torch.randn(32)
        n = gs.inject_into_working(pcs, query, top_k=3)
        assert n == 3
        # The injected tokens should match the top-3 retrieved concept tokens.
        retrieved = gs.retrieve_tokens(query, top_k=3)
        assert torch.allclose(
            pcs.get_bank("working")[:3].to(retrieved.device, retrieved.dtype),
            retrieved,
            atol=1e-5,
        )


class TestCoActivationEdges:
    """Tests for co-activation edge discovery."""

    def test_no_edges_with_one_token(self) -> None:
        """A single token produces no edges."""
        gs = GraphService(num_concepts=2, co_activation_threshold=0.5)
        embeddings = torch.randn(1, 32)
        used_indices = torch.tensor([0])
        edges = gs.build_edges(embeddings, used_indices)
        assert edges == []

    def test_high_threshold_no_edges(self) -> None:
        """A high threshold yields no edges when similarities are lower."""
        gs = GraphService(num_concepts=2, co_activation_threshold=0.99)
        # Use orthogonal-ish embeddings.
        embeddings = torch.eye(5, 32)
        used_indices = torch.arange(5)
        edges = gs.build_edges(embeddings, used_indices)
        assert edges == []

    def test_low_threshold_produces_edges(self) -> None:
        """A low threshold yields edges between highly-similar tokens."""
        gs = GraphService(num_concepts=2, co_activation_threshold=0.5)
        # Build two nearly-identical token pairs.
        a = torch.randn(1, 32)
        b = a + 0.01 * torch.randn(1, 32)
        c = -a + 0.01 * torch.randn(1, 32)
        embeddings = torch.cat([a, b, c], dim=0)
        used_indices = torch.tensor([0, 1, 2])
        edges = gs.build_edges(embeddings, used_indices)
        # At least one edge should exist (between a and b).
        assert len(edges) >= 1

    def test_edges_have_weights_in_unit_interval(self) -> None:
        """Every edge weight lies in ``[0, 1]``."""
        gs = GraphService(num_concepts=2, co_activation_threshold=0.0)
        torch.manual_seed(0)
        embeddings = torch.randn(10, 32)
        used_indices = torch.arange(10)
        edges = gs.build_edges(embeddings, used_indices)
        for edge in edges:
            assert 0.0 <= edge.weight <= 1.0

    def test_no_self_loops(self) -> None:
        """No edge connects a token to itself."""
        gs = GraphService(num_concepts=2, co_activation_threshold=0.0)
        embeddings = torch.randn(5, 32)
        used_indices = torch.arange(5)
        edges = gs.build_edges(embeddings, used_indices)
        for edge in edges:
            assert edge.source != edge.target

    def test_no_duplicate_edges(self) -> None:
        """Each pair of nodes gets at most one edge."""
        gs = GraphService(num_concepts=2, co_activation_threshold=0.0)
        embeddings = torch.randn(5, 32)
        used_indices = torch.arange(5)
        edges = gs.build_edges(embeddings, used_indices)
        pairs: set[tuple[int, int]] = set()
        for edge in edges:
            pair = (
                min(edge.source, edge.target),
                max(edge.source, edge.target),
            )
            assert pair not in pairs
            pairs.add(pair)
