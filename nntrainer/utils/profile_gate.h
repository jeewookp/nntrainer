// SPDX-License-Identifier: Apache-2.0
/**
 * @file   profile_gate.h
 * @brief  Single env-gated switch shared by every prefill-only
 *         PROFILE block destructor.  Set
 *         NNTRAINER_SUPPRESS_PREFILL_PROFILE=1 to silence them while
 *         iterating on decode; leave unset to see the usual output.
 *
 *         Called once per process at static-storage teardown, so the
 *         local-static getenv read is cheap.
 */

#ifndef __NNTRAINER_PROFILE_GATE_H__
#define __NNTRAINER_PROFILE_GATE_H__

#include <cstdlib>

namespace nntrainer {

inline bool prefill_profile_suppressed() {
  static const bool s =
    std::getenv("NNTRAINER_SUPPRESS_PREFILL_PROFILE") != nullptr;
  return s;
}

} // namespace nntrainer

#endif // __NNTRAINER_PROFILE_GATE_H__
