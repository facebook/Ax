/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace py3 ax.storage.future.thrift

struct GenerationNode {
  1: string name;
}

struct GenerationStrategy {
  1: string name;
  2: list<GenerationNode> nodes;
  3: i32 current_node_index;
}
