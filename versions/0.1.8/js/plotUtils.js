/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// helper functions used across multiple plots
function rgb(rgb_array) {
  return 'rgb(' + rgb_array.join() + ')';
}

function copy_and_reverse(arr) {
  const copy = arr.slice();
  copy.reverse();
  return copy;
}

function axis_range(grid, is_log) {
  return is_log ?
    [Math.log10(Math.min(...grid)), Math.log10(Math.max(...grid))]:
    [Math.min(...grid), Math.max(...grid)];
}

function relativize_data(f, sd, rel, arm_data, metric) {
  // if relative, extract status quo & compute ratio
  const f_final = rel === true ? [] : f;
  const sd_final = rel === true ? []: sd;

  if (rel === true) {
    const f_sq = (
      arm_data['in_sample'][arm_data['status_quo_name']]['y'][metric]
    );
    const sd_sq = (
      arm_data['in_sample'][arm_data['status_quo_name']]['se'][metric]
    );

    for (let i = 0; i < f.length; i++) {
      res = relativize(f[i], sd[i], f_sq, sd_sq);
      f_final.push(100 * res[0]);
      sd_final.push(100 * res[1]);
    }
  }

  return [f_final, sd_final];
}

function relativize(m_t, sem_t, m_c, sem_c) {
  r_hat = (
    (m_t - m_c) / Math.abs(m_c) -
    Math.pow(sem_c, 2) * m_t / Math.pow(Math.abs(m_c), 3)
  );
  variance = (
    (Math.pow(sem_t, 2) + Math.pow((m_t / m_c * sem_c), 2)) /
    Math.pow(m_c, 2)
   )
   return [r_hat, Math.sqrt(variance)];
}
