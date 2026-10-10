"""Concept graph over long-term memory.

The graph is a background knowledge-organisation structure: the transformer
never attends to it. Concepts are clusters of long-term memory embeddings,
edges join co-activated memory pairs, and retrieval projects the concepts
closest to a query back into memory tokens that are injected into working
memory for the next reasoning pass.

How closeness is measured is a `Metric`. Cosine and Euclidean ship here; a
new metric is one subclass registered with `register`, and nothing else in
this module changes:

    @graph.register("manhattan")
    class Manhattan(graph.Metric):
        ...

    graph.Cluster(8, metric="manhattan")
"""

import abc
import dataclasses
from collections.abc import Callable

import torch
from torch.nn import functional

from ucsa.models import cognitive as pcs_state


class Metric(abc.ABC):
    """How points are compared, and how a group of points is summarised.

    A metric defines three things that k-means needs: the preprocessing of
    raw points, a distance (smaller is closer), and the centre of a group.
    It also defines a similarity (larger is closer, 1 for identical points)
    that edge thresholds and retrieval use.
    """

    name: str = ""

    def prepare(self, points: torch.Tensor) -> torch.Tensor:
        """Maps raw points into the space the metric compares in.

        Args:
          points: Raw points of shape `(n, dim)`.

        Returns:
          Prepared points of the same shape. The default is the identity.
        """
        return points

    @abc.abstractmethod
    def distance(
        self, points: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """Returns pairwise distances, smaller meaning closer.

        Args:
          points: Prepared points of shape `(n, dim)`.
          centers: Prepared centres of shape `(k, dim)`.

        Returns:
          Distances of shape `(n, k)`.
        """

    @abc.abstractmethod
    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns pairwise similarities, larger meaning closer.

        Args:
          a: Prepared points of shape `(n, dim)`.
          b: Prepared points of shape `(m, dim)`.

        Returns:
          Similarities of shape `(n, m)`; identical points score 1.
        """

    @abc.abstractmethod
    def center(self, members: torch.Tensor) -> torch.Tensor:
        """Returns the centre of a non-empty group of prepared points.

        Args:
          members: Prepared points of shape `(n, dim)`.

        Returns:
          The centre, of shape `(dim,)`.
        """


METRICS: dict[str, type[Metric]] = {}


def register(name: str) -> Callable[[type[Metric]], type[Metric]]:
    """Class decorator that adds a `Metric` to the registry.

    Args:
      name: Key to look the metric up by.

    Returns:
      A decorator returning the class unchanged.

    Raises:
      ValueError: If `name` is empty or already registered.
    """

    def decorator(cls: type[Metric]) -> type[Metric]:
        if not name:
            raise ValueError("metric name must be non-empty")
        if name in METRICS:
            raise ValueError(f"metric {name!r} is already registered")
        cls.name = name
        METRICS[name] = cls
        return cls

    return decorator


def resolve(spec: str | Metric) -> Metric:
    """Turns a metric name or instance into a `Metric` instance.

    Args:
      spec: A registered name or a ready instance.

    Returns:
      The metric.

    Raises:
      ValueError: If a name is not registered.
    """
    if isinstance(spec, Metric):
        return spec
    if spec not in METRICS:
        raise ValueError(f"unknown metric {spec!r}; known: {sorted(METRICS)}")
    return METRICS[spec]()


@register("cosine")
class Cosine(Metric):
    """Angle between unit vectors; magnitude is ignored."""

    def prepare(self, points: torch.Tensor) -> torch.Tensor:
        """Normalises points to unit length."""
        return functional.normalize(points, p=2, dim=-1)

    def distance(
        self, points: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """Returns `1 - cosine similarity`."""
        return 1.0 - points @ centers.T

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the cosine similarity."""
        return a @ b.T

    def center(self, members: torch.Tensor) -> torch.Tensor:
        """Returns the mean direction as a unit vector."""
        return functional.normalize(members.mean(dim=0), p=2, dim=-1)


@register("euclidean")
class Euclidean(Metric):
    """Straight-line distance; magnitude matters."""

    def distance(
        self, points: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """Returns the L2 distance."""
        return torch.cdist(points, centers)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns `1 / (1 + distance)`, which is 1 for identical points."""
        return 1.0 / (1.0 + torch.cdist(a, b))

    def center(self, members: torch.Tensor) -> torch.Tensor:
        """Returns the mean point."""
        return members.mean(dim=0)


@dataclasses.dataclass
class Clustering:
    """Result of `Cluster.fit`.

    Attributes:
      assignments: Cluster id per point, shape `(n,)`. Ids that never occur
        are empty clusters.
      centers: Cluster centres, shape `(num_clusters, dim)`; centres of
        empty clusters are unspecified but finite.
    """

    assignments: torch.Tensor
    centers: torch.Tensor


class Cluster:
    """K-means clustering under a pluggable `Metric`.

    Deterministic for a given seed and input; implemented in pure torch.
    """

    def __init__(
        self,
        num_clusters: int,
        metric: str | Metric = "cosine",
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        seed: int = 0,
    ) -> None:
        """Initialises the clusterer.

        Args:
          num_clusters: Number of clusters.
          metric: Registered metric name or a `Metric` instance.
          max_iterations: Maximum Lloyd iterations.
          tolerance: Stop when no centre moves more than this.
          seed: Seed for the initial centres.

        Raises:
          ValueError: If a count is not positive or the metric is unknown.
        """
        if num_clusters <= 0:
            raise ValueError(
                f"num_clusters must be positive, got {num_clusters}."
            )
        if max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {max_iterations}."
            )
        self.num_clusters = num_clusters
        self.metric = resolve(metric)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed

    def fit(self, points: torch.Tensor) -> Clustering:
        """Clusters `points`.

        Args:
          points: Tensor of shape `(n, dim)`.

        Returns:
          The assignments and centres.

        Raises:
          ValueError: If `points` is not 2-D or contains NaN or infinity.
        """
        if points.dim() != 2:
            raise ValueError(
                f"points must be 2D, got shape {tuple(points.shape)}."
            )
        if not torch.isfinite(points).all():
            raise ValueError("points contain NaN or infinity.")
        count, dim = points.shape
        k = self.num_clusters
        if count == 0:
            return Clustering(
                torch.empty(0, dtype=torch.long), torch.zeros(k, dim)
            )
        prepared = self.metric.prepare(points)
        if count <= k:  # Every point is its own cluster; spare centres are 0.
            centers = torch.zeros(k, dim)
            centers[:count] = prepared
            return Clustering(torch.arange(count, dtype=torch.long), centers)
        generator = torch.Generator().manual_seed(self.seed)
        centers = prepared[torch.randperm(count, generator=generator)[:k]]
        centers = centers.clone()
        assignments = torch.full((count,), -1, dtype=torch.long)
        for _ in range(self.max_iterations):
            nearest = self.metric.distance(prepared, centers).argmin(dim=-1)
            if torch.equal(nearest, assignments):
                break
            assignments = nearest
            updated = centers.clone()  # An empty cluster keeps its centre.
            for cluster_id in range(k):
                members = prepared[assignments == cluster_id]
                if len(members):
                    updated[cluster_id] = self.metric.center(members)
            shift = (centers - updated).norm(dim=-1).max().item()
            centers = updated
            if shift < self.tolerance:
                break
        return Clustering(assignments, centers)


@dataclasses.dataclass
class Concept:
    """A concept extracted from long-term memory.

    Attributes:
      centroid: Concept centre, shape `(dim,)`.
      member_indices: Long-term slots assigned to the concept.
      token: Memory token injected into working memory, shape `(dim,)`.
    """

    centroid: torch.Tensor
    member_indices: list[int] = dataclasses.field(default_factory=list)
    token: torch.Tensor | None = None


@dataclasses.dataclass
class Edge:
    """A weighted link between two co-activated long-term slots.

    Attributes:
      source: Source long-term index.
      target: Target long-term index.
      weight: Similarity of the pair under the graph's metric.
    """

    source: int
    target: int
    weight: float


@dataclasses.dataclass
class Memory:
    """A graph built from a state's long-term bank.

    Attributes:
      concepts: Discovered concepts.
      edges: Co-activation edges.
    """

    concepts: list[Concept] = dataclasses.field(default_factory=list)
    edges: list[Edge] = dataclasses.field(default_factory=list)


class Graph:
    """Builds a concept graph from a state and answers queries against it."""

    def __init__(
        self,
        num_concepts: int = 16,
        threshold: float = 0.5,
        max_iterations: int = 20,
        metric: str | Metric = "cosine",
    ) -> None:
        """Initialises the graph builder.

        Args:
          num_concepts: Number of concept clusters.
          threshold: Minimum similarity for two slots to be linked.
          max_iterations: Lloyd iterations for the clustering.
          metric: Registered metric name or a `Metric` instance.

        Raises:
          ValueError: If `num_concepts` is not positive, `threshold` is
            outside `[0, 1]`, or the metric is unknown.
        """
        if num_concepts <= 0:
            raise ValueError(
                f"num_concepts must be positive, got {num_concepts}."
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
        self.num_concepts = num_concepts
        self.threshold = threshold
        self.metric = resolve(metric)
        self.cluster = Cluster(
            num_concepts, self.metric, max_iterations=max_iterations
        )
        self.memory = Memory()

    def build(self, state: pcs_state.State) -> Memory:
        """Builds the graph from the used slots of the long-term bank.

        Args:
          state: Current cognitive state.

        Returns:
          The new `Memory`, also stored as `self.memory`.
        """
        long_term = state.get_bank("long_term")
        used = torch.nonzero(
            state.metadata("long_term", "usage") > 0, as_tuple=False
        ).squeeze(-1)
        if used.numel() == 0:
            self.memory = Memory()
            return self.memory
        embeddings = long_term[used]
        clustering = self.cluster.fit(embeddings)
        concepts = []
        for cluster_id in range(self.num_concepts):
            mask = clustering.assignments == cluster_id
            if not mask.any():
                continue
            centroid = clustering.centers[cluster_id]
            concepts.append(
                Concept(
                    centroid=centroid,
                    member_indices=[int(i) for i in used[mask].tolist()],
                    token=self.token(centroid, embeddings[mask]),
                )
            )
        self.memory = Memory(concepts, self.connect(embeddings, used))
        return self.memory

    def token(
        self, centroid: torch.Tensor, members: torch.Tensor
    ) -> torch.Tensor:
        """Builds the memory token for a concept.

        The centroid plus a small share of the mean member residual, which
        keeps some of the within-concept spread.

        Args:
          centroid: Concept centre of shape `(dim,)`.
          members: Member embeddings of shape `(n, dim)`.

        Returns:
          A detached token of shape `(dim,)`.
        """
        return (centroid + 0.1 * (members.mean(dim=0) - centroid)).detach()

    def connect(
        self, embeddings: torch.Tensor, indices: torch.Tensor
    ) -> list[Edge]:
        """Links every pair whose similarity reaches the threshold.

        Args:
          embeddings: Used slot embeddings of shape `(n, dim)`.
          indices: Long-term index of each embedding, shape `(n,)`.

        Returns:
          One edge per qualifying pair, with `source < target` by position.
        """
        if embeddings.shape[0] < 2:
            return []
        prepared = self.metric.prepare(embeddings)
        similarity = self.metric.similarity(prepared, prepared)
        rows, cols = torch.triu_indices(*similarity.shape, offset=1)
        weights = similarity[rows, cols]
        keep = weights >= self.threshold
        return [
            Edge(int(indices[i]), int(indices[j]), float(w))
            for i, j, w in zip(
                rows[keep].tolist(),
                cols[keep].tolist(),
                weights[keep].tolist(),
                strict=True,
            )
        ]

    def retrieve(self, query: torch.Tensor, top_k: int = 4) -> list[Concept]:
        """Returns the concepts most similar to `query`.

        Args:
          query: Tensor of shape `(dim,)` or `(batch, dim)`; only the first
            row of a batch is used.
          top_k: Maximum number of concepts.

        Returns:
          Up to `top_k` concepts, most similar first; empty before `build`.
        """
        if not self.memory.concepts or top_k <= 0:
            return []
        if query.dim() == 2:
            query = query[0]
        centroids = torch.stack([c.centroid for c in self.memory.concepts])
        scores = self.metric.similarity(
            self.metric.prepare(query.unsqueeze(0)),
            self.metric.prepare(centroids),
        ).squeeze(0)
        top = torch.topk(scores, min(top_k, scores.shape[0])).indices
        return [self.memory.concepts[int(i)] for i in top.tolist()]

    def retrieve_tokens(
        self, query: torch.Tensor, top_k: int = 4
    ) -> torch.Tensor:
        """Returns the tokens of the concepts most similar to `query`.

        Args:
          query: Tensor of shape `(dim,)` or `(batch, dim)`.
          top_k: Maximum number of tokens.

        Returns:
          Tensor of shape `(n, dim)` with `n <= top_k`; `(0, dim)` if none.
        """
        concepts = self.retrieve(query, top_k)
        if not concepts:
            return torch.zeros(0, query.shape[-1])
        return torch.stack(
            [c.token if c.token is not None else c.centroid for c in concepts]
        )

    def inject(
        self,
        state: pcs_state.State,
        query: torch.Tensor,
        top_k: int = 4,
    ) -> int:
        """Writes retrieved concept tokens into the front of working memory.

        Args:
          state: State whose working bank is updated in place.
          query: Tensor of shape `(dim,)` or `(batch, dim)`.
          top_k: Maximum number of tokens to inject.

        Returns:
          The number of tokens written.
        """
        tokens = self.retrieve_tokens(query, top_k)
        working = state.get_bank("working")
        count = min(tokens.shape[0], working.shape[0])
        if count == 0:
            return 0
        with torch.no_grad():
            working[:count] = tokens[:count].to(
                device=working.device, dtype=working.dtype
            )
        return count
