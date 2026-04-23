// context_filter.cc

#include "context_filter.h"

#include <grammarinator/runtime/Rule.hpp>

namespace mlir_fuzzer {

using grammarinator::runtime::Rule;

// ---------------------------------------------------------------------------
// Ancestor check
// ---------------------------------------------------------------------------
//
// Walk up from recipient and donor simultaneously for up to k_ancestors steps.
//
//   - Both null at the same step   → donor ran out, break and return true.
//   - Donor null, recipient not    → donor ran out, break and return true.
//   - Recipient null, donor not    → recipient can't satisfy donor, return
//   false.
//   - Both non-null but names differ → mismatch, return false.

bool ContextFilter::verify_ancestors(const Rule *recipient,
                                     const Rule *donor) const {
  const Rule *r = recipient->parent;
  const Rule *d = donor->parent;

  for (int i = 0; i < k_ancestors; ++i) {
    if (!d) {
      // Donor ran out — recipient satisfies all donor requirements.
      break;
    }
    if (!r) {
      // Recipient ran out while donor still has ancestors.
      return false;
    }
    if (r->name != d->name) {
      return false;
    }
    r = r->parent;
    d = d->parent;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Left sibling check
// ---------------------------------------------------------------------------
//
// Walk left from recipient and donor simultaneously for up to l_siblings steps.
// Same asymmetric null logic as ancestors.

bool ContextFilter::verify_left_siblings(const Rule *recipient,
                                         const Rule *donor) const {
  const Rule *r = recipient->left_sibling();
  const Rule *d = donor->left_sibling();

  for (int i = 0; i < l_siblings; ++i) {
    if (!d) {
      break;
    }
    if (!r) {
      return false;
    }
    if (r->name != d->name) {
      return false;
    }
    r = r->left_sibling();
    d = d->left_sibling();
  }
  return true;
}

// ---------------------------------------------------------------------------
// Right sibling check
// ---------------------------------------------------------------------------
//
// Walk right from recipient and donor simultaneously for up to r_siblings
// steps. Same asymmetric null logic.

bool ContextFilter::verify_right_siblings(const Rule *recipient,
                                          const Rule *donor) const {
  const Rule *r = recipient->right_sibling();
  const Rule *d = donor->right_sibling();

  for (int i = 0; i < r_siblings; ++i) {
    if (!d) {
      break;
    }
    if (!r) {
      return false;
    }
    if (r->name != d->name) {
      return false;
    }
    r = r->right_sibling();
    d = d->right_sibling();
  }
  return true;
}

// ---------------------------------------------------------------------------
// Top-level verify
// ---------------------------------------------------------------------------

bool ContextFilter::verify(const Rule *recipient, const Rule *donor) const {
  return verify_ancestors(recipient, donor) &&
         verify_left_siblings(recipient, donor) &&
         verify_right_siblings(recipient, donor);
}

} // namespace mlir_fuzzer
