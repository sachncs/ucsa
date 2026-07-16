"""Graph memory service.

Graph memory is a *background knowledge organization system*. The graph
itself is never attended by the transformer. Concepts are clusters of
long-term memory embeddings discovered via cosine k-means (in pure
torch). Edges are placed between co-activated memory pairs. Retrieval
projects relevant concept nodes back into memory tokens, which the
:class:`ucsa.models.memory_service.MemoryService` injects into Working
Memory at the start of the next reasoning pass.

The service is decoupled from the operator: it observes PCS state and
emits memory tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from ucsa.models.state import BANK_NAMES, PersistentCognitiveState

LOGGER = logging.getLogger(__name__)


@dataclass
class Concept:
    """A concept extracted from long-term memory.

    Attributes:
        centroid: Tensor of shape ``(hidden_size,)`` representing the
            concept's centroid in embedding space.
        member_indices: Indices of long-term memory slots assigned to
            this concept.
        token: Optional pre-built memory token of shape ``(hidden_size,)``
            used when injecting the concept back into the PCS.
    """

    centroid: Tensor
    member_indices: list[int] = field(default_factory=list)
    token: Tensor | None = None


@dataclass
class GraphEdge:
    """A weighted edge between two memory tokens.

    Attributes:
        source: Source long-term index.
        target: Target long-term index.
        weight: Edge weight in ``[0, 1]``.
    """

    source: int
    target: int
    weight: float


@dataclass
class GraphMemory:
    """A graph memory built from a PCS's long-term bank.

    Attributes:
        concepts: Discovered concepts.
        edges: Co-activation edges.
    """

    concepts: list[Concept] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


class CosineClusterer:
    """Cosine k-means clusterer implemented in pure torch.

    The clusterer normalises embeddings to unit length and runs Lloyd's
    algorithm for a small fixed number of iterations. It is intentionally
    simple: it does not need to be production-grade, just deterministic
    and dependency-free.
    """

    def __init__(
        self,
        num_clusters: int,
        max_iterations: int = 20,
        tolerance: float = 1e-4,
    ) -> None:
        """Initialise the clusterer.

        Args:
            num_clusters: Target number of clusters.
            max_iterations: Maximum Lloyd iterations.
            tolerance: Convergence threshold on centroid movement.
        """
        if num_clusters <= 0:
            raise ValueError(
                f"num_clusters must be positive, got {num_clusters}."
            )
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def cluster(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Cluster ``embeddings`` into ``num_clusters`` clusters.

        Args:
            embeddings: Tensor of shape ``(n, hidden_size)``.

        Returns:
            Tuple ``(assignments, centroids)``. ``assignments`` has shape
            ``(n,)`` with one cluster id per embedding; ``centroids`` has
            shape ``(num_clusters, hidden_size)``.
        """
        if embeddings.dim() != 2:
            raise ValueError(
                f"embeddings must be 2D, got shape {tuple(embeddings.shape)}."
            )
        n = embeddings.shape[0]
        if n == 0:
            empty_assignments = torch.empty(0, dtype=torch.long)
            empty_centroids = torch.zeros(
                self.num_clusters, embeddings.shape[-1]
            )
            return empty_assignments, empty_centroids
        normalised = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        if n <= self.num_clusters:
            # Degenerate case: each point is its own cluster; pad with zeros.
            centroids = torch.zeros(self.num_clusters, embeddings.shape[-1])
            centroids[:n] = normalised
            assignments = torch.arange(n, dtype=torch.long)
            # Assign surplus clusters to existing points in round-robin.
            for extra in range(n, self.num_clusters):
                assignments = torch.cat(
                    [assignments, torch.tensor([extra % n], dtype=torch.long)]
                )
            return assignments, centroids
        # Random initial centroids drawn from the input.
        generator = torch.Generator()
        generator.manual_seed(0)
        init_indices = torch.randperm(n, generator=generator)[: self.num_clusters]
        centroids = normalised[init_indices].clone()
        assignments = torch.zeros(n, dtype=torch.long)
        for _ in range(self.max_iterations):
            similarity = normalised @ centroids.T
            new_assignments = similarity.argmax(dim=-1)
            if torch.equal(new_assignments, assignments):
                break
            assignments = new_assignments
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.num_clusters)
            for cluster_id in range(self.num_clusters):
                mask = assignments == cluster_id
                if mask.any():
                    new_centroids[cluster_id] = normalised[mask].mean(dim=0)
                    counts[cluster_id] = float(mask.sum().item())
            new_centroids = torch.nn.functional.normalize(
                new_centroids, p=2, dim=-1
            )
            shift = (centroids - new_centroids).norm(dim=-1).max().item()
            centroids = new_centroids
            if shift < self.tolerance:
                break
        return assignments, centroids


class GraphService:
    """Build and query the graph memory."""

    def __init__(
        self,
        num_concepts: int = 16,
        co_activation_threshold: float = 0.5,
        max_iterations: int = 20,
    ) -> None:
        """Initialise the graph service.

        Args:
            num_concepts: Number of concept clusters.
            co_activation_threshold: Minimum cosine similarity to declare
                two memory tokens co-activated.
            max_iterations: Lloyd iterations for the clusterer.
        """
        if num_concepts <= 0:
            raise ValueError(
                f"num_concepts must be positive, got {num_concepts}."
            )
        if not 0.0 <= co_activation_threshold <= 1.0:
            raise ValueError(
                f"co_activation_threshold must be in [0, 1], "
                f"got {co_activation_threshold}."
            )
        self.num_concepts = num_concepts
        self.co_activation_threshold = co_activation_threshold
        self.clusterer = CosineClusterer(
            num_clusters=num_concepts,
            max_iterations=max_iterations,
        )
        self.last_graph: GraphMemory = GraphMemory()

    def build_graph(self, cstate: PersistentCognitiveState) -> GraphMemory:
        """Build a graph from the PCS's long-term bank.

        Args:
            cstate: The current PCS.

        Returns:
            The constructed :class:`GraphMemory`.
        """
        long_term = cstate.get_bank("long_term")
        usage = getattr(cstate, "meta_usage_long_term")
        used_mask = usage > 0
        if not used_mask.any():
            self.last_graph = GraphMemory()
            return self.last_graph
        used_indices = torch.nonzero(used_mask, as_tuple=False).squeeze(-1)
        used_embeddings = long_term[used_indices]
        assignments, centroids = self.clusterer.cluster(used_embeddings)
        concepts: list[Concept] = []
        for cluster_id in range(self.num_concepts):
            mask = assignments == cluster_id
            if not mask.any():
                continue
            member_used_indices = used_indices[mask]
            centroid = centroids[cluster_id]
            token = self._build_concept_token(centroid, used_embeddings[mask])
            concepts.append(
                Concept(
                    centroid=centroid,
                    member_indices=[int(i) for i in member_used_indices.tolist()],
                    token=token,
                )
            )
        edges = self._build_edges(used_embeddings, used_indices)
        self.last_graph = GraphMemory(concepts=concepts, edges=edges)
        return self.last_graph

    def _build_concept_token(
        self, centroid: Tensor, member_embeddings: Tensor
    ) -> Tensor:
        """Build the concept's memory token.

        Args:
            centroid: Cosine-normalised centroid of shape ``(hidden,)``.
            member_embeddings: Embeddings of member tokens
                ``(num_members, hidden)``.

        Returns:
            Token of shape ``(hidden,)``.
        """
        # Combine centroid and mean of (member - centroid) residuals to
        # preserve variance information.
        residual = member_embeddings.mean(dim=0) - centroid
        return (centroid + 0.1 * residual).detach()

    def _build_edges(
        self,
        used_embeddings: Tensor,
        used_indices: Tensor,
    ) -> list[GraphEdge]:
        """Build co-activation edges.

        Args:
            used_embeddings: Embeddings of used memory tokens.
            used_indices: Original long-term indices corresponding to
                ``used_embeddings``.

        Returns:
            List of edges with weight above the configured threshold.
        """
        if used_embeddings.shape[0] < 2:
            return []
        normalised = torch.nn.functional.normalize(
            used_embeddings, p=2, dim=-1
        )
        similarity = normalised @ normalised.T
        # Zero out the diagonal.
        similarity.fill_diagonal_(0.0)
        above = similarity >= self.co_activation_threshold
        # Only emit edges once per pair (i < j).
        edges: list[GraphEdge] = []
        for i in range(above.shape[0]):
            for j in range(i + 1, above.shape[1]):
                if above[i, j]:
                    weight = float(similarity[i, j].item())
                    edges.append(
                        GraphEdge(
                            source=int(used_indices[i].item()),
                            target=int(used_indices[j].item()),
                            weight=weight,
                        )
                    )
        return edges

    def retrieve(
        self,
        query: Tensor,
        top_k: int = 4,
    ) -> list[Concept]:
        """Retrieve the ``top_k`` most relevant concepts for ``query``.

        Args:
            query: Tensor of shape ``(hidden,)`` or ``(batch, hidden)``.
            top_k: Number of concepts to return.

        Returns:
            Up to ``top_k`` :class:`Concept` objects sorted by similarity.
        """
        if not self.last_graph.concepts:
            return []
        if query.dim() == 2:
            query = query[0]
        query_norm = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=-1)
        centroids = torch.stack(
            [concept.centroid for concept in self.last_graph.concepts],
            dim=0,
        )
        centroids_norm = torch.nn.functional.normalize(centroids, p=2, dim=-1)
        similarities = (query_norm @ centroids_norm.T).squeeze(0)
        k_eff = min(top_k, similarities.shape[0])
        if k_eff <= 0:
            return []
        _, top_indices = torch.topk(similarities, k_eff, largest=True)
        return [self.last_graph.concepts[int(i)] for i in top_indices.tolist()]

    def retrieve_tokens(
        self,
        query: Tensor,
        top_k: int = 4,
    ) -> Tensor:
        """Retrieve concept tokens for ``query``.

        Args:
            query: Tensor of shape ``(hidden,)`` or ``(batch, hidden)``.
            top_k: Number of tokens to return.

        Returns:
            Tensor of shape ``(top_k, hidden)``.
        """
        concepts = self.retrieve(query, top_k=top_k)
        if not concepts:
            hidden = query.shape[-1]
            return torch.zeros(0, hidden)
        return torch.stack(
            [
                concept.token if concept.token is not None else concept.centroid
                for concept in concepts
            ],
            dim=0,
        )


__all__ = [
    "Concept",
    "CosineClusterer",
    "GraphEdge",
    "GraphMemory",
    "GraphService",
]


def supported_banks() -> Sequence[str]:
    """internal: list banks whose embeddings are eligible for clustering."""
    return list(BANK_NAMES)