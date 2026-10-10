"""Properties of the concept graph: determinism, failure handling, extension.

The tests are grouped by the guarantee they establish, not by method:
deterministic data flow, rejection of bad input, recovery after a failure,
resilience to degenerate data, extension through new metrics, and scale.
"""

import time

import pytest
import torch

from ucsa.models import cognitive as pcs_state, graph

HIDDEN = 16


def make_state(points: torch.Tensor | None = None):
    """Returns a state whose long-term bank holds `points` (all marked used)."""
    state = pcs_state.State(pcs_state.Config(hidden_size=HIDDEN))
    if points is not None:
        with torch.no_grad():
            state.get_bank("long_term")[: len(points)] = points
            state.metadata("long_term", "usage")[: len(points)] = 1.0
    return state


def blobs(per_blob=10, spread=0.05, seed=0):
    """Three well-separated groups in different directions."""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.eye(HIDDEN)[:3] * 5.0
    pts = [
        c + spread * torch.randn(per_blob, HIDDEN, generator=gen)
        for c in centers
    ]
    return torch.cat(pts)


class Manhattan(graph.Metric):
    """L1 distance, defined only here: proves new metrics need no edits."""

    def distance(self, points, centers):
        return torch.cdist(points, centers, p=1.0)

    def similarity(self, a, b):
        return 1.0 / (1.0 + torch.cdist(a, b, p=1.0))

    def center(self, members):
        return members.median(dim=0).values


# ---------------------------------------------------------------- determinism


class TestDeterministicFlow:
    def test_same_input_gives_identical_clustering(self):
        pts = blobs()
        a = graph.Cluster(3).fit(pts)
        b = graph.Cluster(3).fit(pts)
        assert torch.equal(a.assignments, b.assignments)
        assert torch.equal(a.centers, b.centers)

    def test_same_state_gives_identical_graph_every_time(self):
        state = make_state(blobs())
        first = graph.Graph(num_concepts=3).build(state)
        second = graph.Graph(num_concepts=3).build(state)
        assert [c.member_indices for c in first.concepts] == [
            c.member_indices for c in second.concepts
        ]
        assert [(e.source, e.target) for e in first.edges] == [
            (e.source, e.target) for e in second.edges
        ]

    def test_well_separated_groups_are_recovered_exactly(self):
        clustering = graph.Cluster(3).fit(blobs(per_blob=10))
        groups = {
            tuple(clustering.assignments[i * 10 : (i + 1) * 10].tolist())
            for i in range(3)
        }
        assert all(len(set(g)) == 1 for g in groups)  # one id per blob
        assert len({g[0] for g in groups}) == 3  # distinct ids

    def test_result_is_a_lloyd_fixed_point(self):
        pts = blobs(spread=0.5)
        cluster = graph.Cluster(3, max_iterations=100)
        out = cluster.fit(pts)
        prepared = cluster.metric.prepare(pts)
        nearest = cluster.metric.distance(prepared, out.centers).argmin(-1)
        assert torch.equal(nearest, out.assignments)

    def test_assignments_are_valid_ids(self):
        out = graph.Cluster(4).fit(torch.randn(50, HIDDEN))
        assert out.assignments.min() >= 0
        assert out.assignments.max() < 4
        assert out.centers.shape == (4, HIDDEN)

    def test_edges_are_unique_ordered_and_above_threshold(self):
        state = make_state(blobs())
        g = graph.Graph(num_concepts=3, threshold=0.9)
        memory = g.build(state)
        pairs = [(e.source, e.target) for e in memory.edges]
        assert len(pairs) == len(set(pairs))
        assert all(s < t for s, t in pairs)
        assert all(e.weight >= 0.9 for e in memory.edges)
        # Same-blob pairs are linked, cross-blob pairs are not.
        assert all(s // 10 == t // 10 for s, t in pairs)

    def test_edge_weights_equal_the_metric_similarity(self):
        pts = blobs()
        g = graph.Graph(num_concepts=3, threshold=0.5)
        prepared = g.metric.prepare(pts)
        sim = g.metric.similarity(prepared, prepared)
        for e in g.connect(pts, torch.arange(len(pts))):
            assert e.weight == pytest.approx(float(sim[e.source, e.target]))

    def test_retrieval_is_sorted_by_similarity(self):
        g = graph.Graph(num_concepts=3)
        g.build(make_state(blobs()))
        query = g.memory.concepts[1].centroid
        found = g.retrieve(query, top_k=3)
        assert found[0] is g.memory.concepts[1]
        scores = [
            float(g.metric.similarity(query[None], c.centroid[None]).detach())
            for c in found
        ]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_tokens_match_retrieved_concepts(self):
        g = graph.Graph(num_concepts=3)
        g.build(make_state(blobs()))
        query = torch.randn(HIDDEN)
        concepts = g.retrieve(query, top_k=2)
        tokens = g.retrieve_tokens(query, top_k=2)
        assert torch.equal(tokens, torch.stack([c.token for c in concepts]))

    def test_inject_writes_front_of_working_in_place(self):
        state = make_state(blobs())
        g = graph.Graph(num_concepts=3)
        g.build(state)
        query = g.memory.concepts[0].centroid
        before = state.get_bank("working").clone()
        n = g.inject(state, query, top_k=2)
        after = state.get_bank("working")
        expected = g.retrieve_tokens(query, top_k=2)
        assert n == 2
        assert torch.allclose(after[:2], expected)
        assert torch.equal(after[2:], before[2:])  # the rest is untouched


# ------------------------------------------------------------------ bad input


class TestBadInput:
    @pytest.mark.parametrize("bad", [0, -3])
    def test_non_positive_cluster_counts_are_rejected(self, bad):
        with pytest.raises(ValueError, match="num_clusters"):
            graph.Cluster(bad)
        with pytest.raises(ValueError, match="num_concepts"):
            graph.Graph(num_concepts=bad)

    def test_non_positive_iterations_are_rejected(self):
        with pytest.raises(ValueError, match="max_iterations"):
            graph.Cluster(2, max_iterations=0)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_threshold_outside_unit_interval_is_rejected(self, bad):
        with pytest.raises(ValueError, match="threshold"):
            graph.Graph(threshold=bad)

    def test_unknown_metric_names_list_the_known_ones(self):
        with pytest.raises(ValueError, match="cosine"):
            graph.Cluster(2, metric="nope")

    @pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
    def test_wrong_rank_is_rejected(self, shape):
        with pytest.raises(ValueError, match="2D"):
            graph.Cluster(2).fit(torch.randn(*shape))

    @pytest.mark.parametrize("poison", [float("nan"), float("inf")])
    def test_non_finite_points_are_rejected(self, poison):
        pts = torch.randn(6, HIDDEN)
        pts[2, 3] = poison
        with pytest.raises(ValueError, match="NaN or infinity"):
            graph.Cluster(2).fit(pts)

    def test_retrieval_with_no_concepts_or_bad_k_returns_nothing(self):
        g = graph.Graph(num_concepts=3)
        assert g.retrieve(torch.randn(HIDDEN)) == []
        g.build(make_state(blobs()))
        assert g.retrieve(torch.randn(HIDDEN), top_k=0) == []
        assert g.retrieve_tokens(torch.randn(HIDDEN), top_k=-1).shape == (
            0,
            HIDDEN,
        )


# ------------------------------------------------------- breakage and recovery


class TestBreakageAndRecovery:
    def test_failed_build_keeps_the_last_good_graph(self):
        g = graph.Graph(num_concepts=3)
        good = g.build(make_state(blobs()))
        poisoned = blobs()
        poisoned[4, 0] = float("nan")
        with pytest.raises(ValueError):
            g.build(make_state(poisoned))
        assert g.memory is good  # not half-overwritten
        assert g.retrieve(torch.randn(HIDDEN))  # still answers queries

    def test_graph_recovers_on_the_next_valid_build(self):
        g = graph.Graph(num_concepts=3)
        g.build(make_state(blobs()))
        bad = blobs()
        bad[0, 0] = float("inf")
        with pytest.raises(ValueError):
            g.build(make_state(bad))
        fresh = g.build(make_state(blobs(seed=7)))
        assert g.memory is fresh
        assert len(fresh.concepts) == 3

    def test_state_is_not_modified_by_a_failed_build(self):
        poisoned = blobs()
        poisoned[1, 1] = float("nan")
        state = make_state(poisoned)
        snapshot = state.get_bank("working").clone()
        with pytest.raises(ValueError):
            graph.Graph(num_concepts=3).build(state)
        assert torch.equal(state.get_bank("working"), snapshot)

    def test_clustering_survives_an_intermediate_empty_cluster(self):
        # Two tight blobs but four clusters requested: some stay empty.
        pts = torch.cat(
            [
                torch.randn(8, HIDDEN) * 0.01 + 1.0,
                torch.randn(8, HIDDEN) * 0.01 - 1.0,
            ]
        )
        out = graph.Cluster(4, metric="euclidean").fit(pts)
        assert torch.isfinite(out.centers).all()
        assert out.assignments.min() >= 0


# ----------------------------------------------------------------- resilience


class TestResilienceToDegenerateData:
    def test_empty_input_gives_empty_clustering(self):
        out = graph.Cluster(3).fit(torch.empty(0, HIDDEN))
        assert out.assignments.numel() == 0
        assert out.centers.shape == (3, HIDDEN)

    def test_fewer_points_than_clusters_gives_one_id_per_point(self):
        out = graph.Cluster(5).fit(torch.randn(2, HIDDEN))
        assert out.assignments.tolist() == [0, 1]
        assert out.centers.shape == (5, HIDDEN)
        assert torch.isfinite(out.centers).all()

    def test_assignments_always_have_one_entry_per_point(self):
        for n in (0, 1, 3, 5, 6, 40):
            out = graph.Cluster(5).fit(torch.randn(n, HIDDEN))
            assert out.assignments.shape == (n,)

    def test_identical_points_form_one_stable_concept(self):
        pts = torch.ones(12, HIDDEN)
        out = graph.Cluster(3).fit(pts)
        assert torch.isfinite(out.centers).all()
        assert len(set(out.assignments.tolist())) == 1

    def test_zero_vectors_do_not_produce_nan(self):
        pts = torch.zeros(10, HIDDEN)
        out = graph.Cluster(2).fit(pts)
        assert torch.isfinite(out.centers).all()

    def test_unused_long_term_gives_an_empty_graph(self):
        memory = graph.Graph().build(make_state(None))
        assert memory.concepts == []
        assert memory.edges == []

    def test_single_used_slot_gives_one_concept_and_no_edges(self):
        memory = graph.Graph(num_concepts=4).build(
            make_state(torch.randn(1, HIDDEN))
        )
        assert len(memory.concepts) == 1
        assert memory.edges == []

    def test_inject_with_nothing_built_is_a_no_op(self):
        state = make_state(None)
        before = state.get_bank("working").clone()
        assert graph.Graph().inject(state, torch.randn(HIDDEN)) == 0
        assert torch.equal(state.get_bank("working"), before)

    def test_inject_is_capped_by_working_capacity(self):
        state = make_state(blobs())
        g = graph.Graph(num_concepts=3)
        g.build(state)
        capacity = state.get_bank("working").shape[0]
        assert g.inject(state, torch.randn(HIDDEN), top_k=10_000) <= min(
            capacity, 3
        )

    def test_batched_query_uses_its_first_row(self):
        g = graph.Graph(num_concepts=3)
        g.build(make_state(blobs()))
        q = torch.randn(HIDDEN)
        batch = torch.stack([q, torch.randn(HIDDEN)])
        assert g.retrieve(batch, 2) == g.retrieve(q, 2)

    def test_no_autograd_graph_is_attached_to_concept_tokens(self):
        g = graph.Graph(num_concepts=3)
        g.build(make_state(blobs().requires_grad_(False)))
        assert all(not c.token.requires_grad for c in g.memory.concepts)


# ------------------------------------------------------ modularity / extension


class TestExtensionThroughMetrics:
    def test_builtin_metrics_are_registered_by_name(self):
        assert {"cosine", "euclidean"} <= set(graph.METRICS)
        assert isinstance(graph.resolve("euclidean"), graph.Euclidean)

    def test_cosine_ignores_magnitude_but_euclidean_does_not(self):
        direction = torch.ones(1, HIDDEN)
        pts = torch.cat(
            [
                direction * 1.0,
                direction * 1.1,
                direction * 10.0,
                direction * 10.1,
            ]
        )
        cosine = graph.Cluster(2, "cosine").fit(pts)
        euclid = graph.Cluster(2, "euclidean").fit(pts)
        # Euclidean separates small from large; cosine sees one direction.
        assert euclid.assignments.tolist() in ([0, 0, 1, 1], [1, 1, 0, 0])
        assert cosine.assignments.tolist() not in ([0, 0, 1, 1], [1, 1, 0, 0])

    def test_euclidean_graph_uses_euclidean_similarity(self):
        g = graph.Graph(num_concepts=2, metric="euclidean", threshold=0.5)
        pts = torch.tensor(
            [[0.0] * HIDDEN, [0.1] + [0.0] * (HIDDEN - 1), [9.0] * HIDDEN]
        )
        edges = g.connect(pts, torch.arange(3))
        assert [(e.source, e.target) for e in edges] == [(0, 1)]
        assert edges[0].weight == pytest.approx(1 / (1 + 0.1), rel=1e-5)

    def test_a_metric_defined_outside_the_module_works_unregistered(self):
        pts = blobs()
        out = graph.Cluster(3, metric=Manhattan()).fit(pts)
        assert out.assignments.max() < 3
        g = graph.Graph(num_concepts=3, metric=Manhattan())
        assert len(g.build(make_state(pts)).concepts) == 3

    def test_registering_a_metric_makes_it_selectable_by_name(
        self, monkeypatch
    ):
        monkeypatch.setattr(graph, "METRICS", dict(graph.METRICS))
        graph.register("manhattan")(Manhattan)
        assert isinstance(graph.resolve("manhattan"), Manhattan)
        assert graph.Cluster(3, "manhattan").metric.name == "manhattan"

    def test_duplicate_and_empty_registrations_are_rejected(self, monkeypatch):
        monkeypatch.setattr(graph, "METRICS", dict(graph.METRICS))
        with pytest.raises(ValueError, match="already registered"):
            graph.register("cosine")(Manhattan)
        with pytest.raises(ValueError, match="non-empty"):
            graph.register("")(Manhattan)

    def test_incomplete_metrics_cannot_be_instantiated(self):
        class Broken(graph.Metric):
            pass

        with pytest.raises(TypeError):
            Broken()


# ---------------------------------------------------------------------- scale


class TestScale:
    def test_many_points_cluster_quickly_with_bounded_output(self):
        pts = torch.randn(5000, 64)
        start = time.time()
        out = graph.Cluster(32, max_iterations=10).fit(pts)
        assert time.time() - start < 30
        assert out.assignments.shape == (5000,)
        assert out.centers.shape == (32, 64)

    def test_iteration_cap_bounds_the_work(self):
        calls = []

        class Counting(graph.Cosine):
            def distance(self, points, centers):
                calls.append(1)
                return super().distance(points, centers)

        graph.Cluster(5, Counting(), max_iterations=3, tolerance=0.0).fit(
            torch.randn(200, HIDDEN)
        )
        assert len(calls) <= 3

    def test_edge_count_is_bounded_by_the_number_of_pairs(self):
        n = 40
        g = graph.Graph(threshold=0.0)
        edges = g.connect(torch.randn(n, HIDDEN), torch.arange(n))
        assert len(edges) <= n * (n - 1) // 2
