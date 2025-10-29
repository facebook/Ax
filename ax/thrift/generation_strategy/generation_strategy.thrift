/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

include "generation_node.thrift"

namespace py3 ax.thrift.generation_strategy // For python-thrift
namespace py ax.thrift.generation_strategy.generation_strategy // For apache thrift

struct GenerationStrategy {
  1: string name;
  2: list<generation_node.GenerationNode> nodes;
  3: i32 current_node_index;
}
