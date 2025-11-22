/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// For python-thrift
namespace py3 ax.thrift.generation_strategy

// For apache thrift
namespace py ax.thrift.generation_strategy.generation_node

struct GenerationNode {
  1: string name;
}
