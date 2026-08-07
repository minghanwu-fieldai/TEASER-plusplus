/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <utility>
#include <vector>

namespace teaser {

/**
 * Connected-component labeling shared by the rotation and translation synchronization stages.
 *
 * The two stages must agree on both the component split and each component's anchor, otherwise the
 * rotation and translation gauges disagree and the composed pose is meaningless. Sharing this makes
 * that structural rather than a convention two files have to remember.
 */
struct SyncComponents {
  /** Component id per node, or -1 for a node that no surviving edge touches. */
  std::vector<int> component;
  /**
   * Node ids per component, ASCENDING. nodes[c][0] is therefore the lowest-indexed node of the
   * component, which is the anchor both stages pin.
   */
  std::vector<std::vector<int>> nodes;
  /** Number of components. Nodes with no edge are not counted as components of their own. */
  int num_components = 0;
};

/**
 * Label connected components over an undirected edge list.
 *
 * Edges referencing out-of-range or negative nodes, and self-loops, are ignored. A node that ends
 * up touched by no edge is deliberately NOT made a singleton component: it gets component -1, and
 * callers report it as unestimated rather than inventing a pose for it. Pass only the edges that
 * survived weighting -- a graph can fragment mid-GNC as correspondences are rejected.
 *
 * @param num_nodes [in] number of nodes; edges must reference 0..num_nodes-1
 * @param edges [in] undirected edges
 * @return the component labeling
 */
SyncComponents labelSyncComponents(int num_nodes, const std::vector<std::pair<int, int>>& edges);

} // namespace teaser
