import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple
from loguru import logger


class TransportGraph:

    def __init__(self):
        self.G = nx.DiGraph()
        self.zone_index: Dict[str, list] = {}

    def build_from_dw(self, topo_df: pd.DataFrame, trafic_df: pd.DataFrame):
        for _, row in topo_df.iterrows():
            self.G.add_node(
                row["arret_id"],
                zone=row.get("zone_nom", "Inconnue"),
                stop_nom=row.get("stop_nom", ""),
            )
            zone = row.get("zone_nom", "Inconnue")
            self.zone_index.setdefault(zone, []).append(row["arret_id"])

        for zone, arrets in self.zone_index.items():
            for i in range(len(arrets) - 1):
                self.G.add_edge(
                    arrets[i], arrets[i + 1],
                    zone=zone,
                    congestion=0.0,
                )

        self._update_weights(trafic_df)
        logger.info(
            f"Graphe : {self.G.number_of_nodes()} noeuds | "
            f"{self.G.number_of_edges()} aretes"
        )

    def _update_weights(self, trafic_df: pd.DataFrame):
        cong_by_zone = trafic_df.groupby("zone_nom")["congestion_index"].mean().to_dict()
        for u, v, data in self.G.edges(data=True):
            zone_u = self.G.nodes[u].get("zone", "")
            self.G[u][v]["congestion"] = cong_by_zone.get(zone_u, 0.0)

    def propagate_delay(self, source_arret: int, delay_seconds: float,
                        max_hops: int = 5) -> Dict[int, float]:
        impacted: Dict[int, float] = {source_arret: delay_seconds}
        queue = [(source_arret, delay_seconds)]
        visited = {source_arret}
        for _ in range(max_hops):
            next_queue = []
            for node, delay in queue:
                for neighbor in self.G.successors(node):
                    if neighbor in visited:
                        continue
                    cong_factor = 1.0 + self.G[node][neighbor].get("congestion", 0.0) * 0.5
                    propagated = delay * 0.7 * cong_factor
                    if propagated < 30:
                        continue
                    impacted[neighbor] = propagated
                    next_queue.append((neighbor, propagated))
                    visited.add(neighbor)
            queue = next_queue
            if not queue:
                break
        return impacted

    def critical_nodes(self, top_n: int = 10) -> pd.DataFrame:
        betweenness = nx.betweenness_centrality(self.G, weight="congestion")
        degree = dict(self.G.degree())
        records = [
            {
                "arret_id":    n,
                "zone":        self.G.nodes[n].get("zone", ""),
                "stop_nom":    self.G.nodes[n].get("stop_nom", ""),
                "betweenness": betweenness.get(n, 0.0),
                "degree":      degree.get(n, 0),
            }
            for n in self.G.nodes()
        ]
        df = pd.DataFrame(records)
        return df.nlargest(top_n, "betweenness").reset_index(drop=True)

    def resilient_path(self, source: int, target: int) -> Tuple[List[int], float]:
        try:
            path = nx.shortest_path(self.G, source, target, weight="congestion")
            cost = sum(
                self.G[path[i]][path[i + 1]].get("congestion", 0)
                for i in range(len(path) - 1)
            )
            return path, cost
        except nx.NetworkXNoPath:
            return [], float("inf")

    def to_pyvis_html(self, output_path: str = "graph.html"):
        from pyvis.network import Network
        net = Network(height="600px", width="100%",
                      bgcolor="#1e293b", font_color="#94a3b8", directed=True)
        for node_id, attrs in self.G.nodes(data=True):
            net.add_node(node_id, label=str(node_id),
                         title=f"Zone: {attrs.get('zone','?')}")
        for u, v, attrs in self.G.edges(data=True):
            net.add_edge(u, v, width=1 + attrs.get("congestion", 0) * 3)
        net.write_html(output_path)
        return output_path