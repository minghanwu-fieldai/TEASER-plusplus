/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <numeric>

#include "sync_components.h"

teaser::SyncComponents
teaser::labelSyncComponents(int num_nodes, const std::vector<std::pair<int, int>>& edges) {
  teaser::SyncComponents out;
  if (num_nodes <= 0) {
    return out;
  }
  out.component.assign(num_nodes, -1);

  std::vector<int> uf(num_nodes);
  std::iota(uf.begin(), uf.end(), 0);
  auto find = [&uf](int x) {
    while (uf[x] != x) {
      uf[x] = uf[uf[x]]; // path halving
      x = uf[x];
    }
    return x;
  };

  std::vector<bool> has_edge(num_nodes, false);
  for (const auto& e : edges) {
    if (e.first < 0 || e.second < 0 || e.first >= num_nodes || e.second >= num_nodes ||
        e.first == e.second) {
      continue;
    }
    has_edge[e.first] = true;
    has_edge[e.second] = true;
    const int ra = find(e.first);
    const int rb = find(e.second);
    if (ra != rb) {
      uf[rb] = ra;
    }
  }

  // Walking nodes in ascending order both numbers the components deterministically and leaves each
  // component's node list sorted, so nodes[c][0] is its lowest-indexed node -- the anchor.
  std::vector<int> root_to_comp(num_nodes, -1);
  for (int i = 0; i < num_nodes; ++i) {
    if (!has_edge[i]) {
      continue;
    }
    const int root = find(i);
    if (root_to_comp[root] < 0) {
      root_to_comp[root] = static_cast<int>(out.nodes.size());
      out.nodes.emplace_back();
    }
    const int comp = root_to_comp[root];
    out.component[i] = comp;
    out.nodes[comp].push_back(i);
  }
  out.num_components = static_cast<int>(out.nodes.size());
  return out;
}
